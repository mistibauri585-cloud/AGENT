import os
import time
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Header

from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import ask_the_principal
from app.services import whisper_service
from app.services import redis_cache
from app.services.rag.pipeline import search_bookshelf, ingest_all_pdfs_from_folder
from app.database import chromadb_client

# Use structured module logger configured globally in main.py
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Fetch environment variables for admin route protection
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")


async def process_core_chat_pipeline(question: str, request_id: str, start_time: float) -> ChatResponse:
    """Shared core orchestration layer enforcing a deterministic routing loop:
    
    Redis Key Validation -> Cache Lookup -> Unified RAG/Web Search -> Groq LLM -> Redis Save.
    """
    normalized_question = question.strip()
    cache_key = None
    
    # 1. Generate & Validate Redis Cache Key Matrix
    try:
        generated_key = redis_cache.generate_cache_key(normalized_question)
        if generated_key:
            cache_key = generated_key
    except Exception as e:
        logger.warning(f"[{request_id}] Redis key generation faulted: {str(e)}")

    # 2. Redis Cache Lookup Execution
    if cache_key:
        try:
            cached_response = redis_cache.get(cache_key)
            if cached_response:
                elapsed_time = round((time.time() - start_time), 3)
                logger.info(f"[{request_id}] Cache Hit | Source: Redis Cache | Time: {elapsed_time}s")
                return ChatResponse(
                    success=True,
                    question=normalized_question,
                    answer=cached_response["answer"],
                    detected_language=cached_response.get("detected_language", "Unknown"),
                    source="Redis Cache",
                    response_time=elapsed_time,
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
        except Exception as e:
            logger.warning(f"[{request_id}] Redis read bypass on exception: {str(e)}")

    # 3. Unified Hybrid Knowledge Base Core Query
    context = ""
    source_identity = "Unknown"
    logger.info(f"[{request_id}] Cache Miss | Initiating unified vector and fallback search workspace.")
    
    try:
        rag_result = search_bookshelf(normalized_question)
        if rag_result and rag_result.get("found"):
            context = rag_result.get("context", "")
            source_identity = rag_result.get("source", "PDF Knowledge Base")
        else:
            context = ""
            source_identity = "No Valid Context Discovered"
    except Exception as e:
        logger.error(f"[{request_id}] Unified RAG pipeline search execution failed: {str(e)}")
        context = ""
        source_identity = "Pipeline Fallback Error Context"

    # 4. Core Language Model Text Synthesis
    try:
        ai_output = ask_the_principal(normalized_question, context)
        detected_lang = ai_output.get("detected_language", "Unknown")
        final_answer = ai_output.get("answer", "")
    except Exception as e:
        logger.error(f"[{request_id}] Core LLM matrix generation crash: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Core processing engine failed to synthesize response: {str(e)}"
        )

    elapsed_time = round((time.time() - start_time), 3)
    logger.info(f"[{request_id}] Pipeline Success | Source: {source_identity} | Time: {elapsed_time}s")

    # 5. Populate Successful Sync Output Back to Validated Redis Key
    if cache_key and final_answer:
        try:
            payload = {"answer": final_answer, "detected_language": detected_lang}
            redis_cache.set(cache_key, payload)
        except Exception as e:
            logger.warning(f"[{request_id}] Redis transaction write failure: {str(e)}")

    return ChatResponse(
        success=True,
        question=normalized_question,
        answer=final_answer,
        detected_language=detected_lang,
        source=source_identity,
        response_time=elapsed_time,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.post("/chat", response_model=ChatResponse)
async def text_chat_endpoint(payload: ChatRequest):
    """Processes entry text interface queries through the centralized RAG processing loop."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    question = payload.question.strip()
    
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question payload cannot be empty or blank structures."
        )
        
    if len(question) > 5000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Question length exceeds maximum supported structure size (5000 chars)."
        )
        
    return await process_core_chat_pipeline(question, request_id, start_time)


@router.post("/voice-chat", response_model=ChatResponse)
async def voice_chat_endpoint(audio_file: UploadFile = File(...)):
    """Receives multi-part audio files, unpackages streams via Groq Whisper APIs,
    and instantly passes strings down to the core cache-RAG pipeline.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    if not audio_file or not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid multipart form binary audio payload required."
        )

    # Validate audio media types before consuming server runtime processing cycles
    allowed_types = {
        "audio/mpeg", "audio/mp3", "audio/wav", 
        "audio/x-wav", "audio/webm", "audio/mp4", "audio/ogg"
    }
    if audio_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format '{audio_file.content_type}'."
        )

    # 1. Audio Transcription Processing with Explicit Signature Dictionary Handlers
    try:
        result = whisper_service.transcribe_audio(audio_file.file, audio_file.filename)
        
        if not result or not result.get("success"):
            error_msg = result.get("message", "Unknown transcription processing error.")
            raise ValueError(error_msg)
            
        question = result.get("transcription", "").strip()
        if not question:
            raise ValueError("Whisper transcription completed but returned an empty text payload string.")
            
    except Exception as e:
        logger.error(f"[{request_id}] Audio transcription layer exception: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audio processing engine failed: {str(e)}"
        )

    # 2. Re-Route into Shared Execution Stream to maintain dry constraints
    response = await process_core_chat_pipeline(question, request_id, start_time)
    
    # Recalculate response latency metrics to fully measure speech integration performance
    response.response_time = round((time.time() - start_time), 3)
    return response


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Aggregates telemetry and status values across interconnected database blocks and interfaces."""
    return {
        "api": "healthy",
        "redis": redis_cache.health_check(),
        "chromadb": chromadb_client.check_database(),
        "whisper": whisper_service.health_check(),
        "llm": {
            "provider": "Groq",
            "model": "qwen/qwen3-32b",
            "status": "healthy"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.post("/reindex", status_code=status.HTTP_200_OK)
def trigger_reindexing(x_admin_token: str = Header(...)):
    """Administrative maintenance endpoint mapping file additions inside the document workspace directory."""
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Invalid or missing administration credentials."
        )

    start_time = time.time()
    try:
        indexing_summary = ingest_all_pdfs_from_folder("pdfs")
        elapsed_time = round((time.time() - start_time), 3)
        return {
            "success": True,
            "indexed_documents": indexing_summary,
            "execution_time": f"{elapsed_time}s",
            "message": "Localized PDF context indexing completed successfully."
        }
    except Exception as e:
        logger.error(f"Administrative vector build task crashed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Asynchronous knowledge reindexing processing failed: {str(e)}"
        )
