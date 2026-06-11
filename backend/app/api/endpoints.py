from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag.pipeline import search_bookshelf, ingest_all_pdfs_from_folder
from app.services.tavily_service import search_the_web
from app.services.gemini_service import ask_the_principal

router = APIRouter(prefix="/api")

@router.post("/chat", response_model=ChatResponse)
async def text_chat_endpoint(payload: ChatRequest):
    # 1. Search local vectors
    rag_result = search_bookshelf(payload.question)
    
    if rag_result["found"]:
        context = rag_result["context"]
        source_type = "PDF Base"
        source_name = rag_result["source"]
    else:
        # 2. Fallback to web search
        context = search_the_web(payload.question)
        source_type = "Web Search"
        source_name = "Tavily Engine"
        
    # 3. Process through language detection and Gemini response engine
    ai_output = ask_the_principal(payload.question, context)
    
    return ChatResponse(
        user_speech_text=payload.question,
        detected_language=ai_output["detected_language"],
        answer=ai_output["answer"],
        source_type=source_type,
        source_name=source_name
    )

@router.post("/voice-chat", response_model=ChatResponse)
async def voice_chat_endpoint(file: UploadFile = File(...)):
    """Accepts raw audio file binaries, runs mock speech transcription for testing, then routes to multi-language agent pipeline."""
    # Production mock entry string for direct client integration verification
    mock_transcription = "मुझे लोन कैसे मिल सकता है और पीएम किसान योजना क्या है?" 
    
    rag_result = search_bookshelf(mock_transcription)
    context = rag_result["context"] if rag_result["found"] else search_the_web(mock_transcription)
    
    ai_output = ask_the_principal(mock_transcription, context)
    
    return ChatResponse(
        user_speech_text=mock_transcription,
        detected_language=ai_output["detected_language"],
        answer=ai_output["answer"],
        source_type="PDF Base" if rag_result["found"] else "Web Search",
        source_name=rag_result["source"] if rag_result["found"] else "Tavily Engine"
    )

@router.get("/health")
def health_check():
    return {"status": "healthy", "system": "Appna Bank AI Backend Online"}

@router.post("/reindex")
def trigger_reindexing():
    message = ingest_all_pdfs_from_folder("pdfs")
    return {"message": message}
