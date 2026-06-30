import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, BinaryIO
import groq

# Reusing the centralized singleton key manager instance across LLM and Whisper
from app.services.key_manager import key_manager  

# Utilize a structured module-level logger configured globally in main.py
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
MODEL_NAME: str = "whisper-large-v3-turbo"

# Content validation constraints
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "webm", "ogg", "flac"}
ALLOWED_MIME_TYPES = {"audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg", "audio/flac", "audio/x-wav"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB Hard ceiling cap boundary

LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
    "ta": "Tamil", "te": "Telugu", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"
}


def connect() -> None:
    """Maintains interface compatibility for lifespan event loops.
    Key management is handled dynamically per request via SupabaseKeyManager.
    """
    logger.info("Whisper Engine cluster connected to SupabaseKeyManager lifecycle rotation.")


def validate_audio(filename: str, file_size_bytes: int, mime_type: Optional[str] = None) -> Dict[str, Any]:
    """Helper implementation to audit structural payload integrity bounds."""
    if not filename:
        return {"valid": False, "message": "Missing file identifier signature metadata."}
        
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False, 
            "message": f"Unsupported audio data format type extension: [.{ext}]. Supported: {list(ALLOWED_EXTENSIONS)}"
        }
        
    if mime_type and mime_type.lower() not in ALLOWED_MIME_TYPES:
        return {
            "valid": False,
            "message": f"Unsupported audio content MIME type detected: [{mime_type}]. Rejected before API transfer."
        }
        
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        return {
            "valid": False, 
            "message": f"Audio block footprint exceeds structural capacity limits. Maximum size budget is {max_mb} MB."
        }
        
    if file_size_bytes <= 0:
        return {"valid": False, "message": "Uploaded audio block cannot contain empty data arrays."}

    return {"valid": True, "message": "Audio file payload validated successfully."}


def transcribe_audio(file_object: BinaryIO, filename: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
    """Transcribes raw binary audio object payloads via Groq's whisper-large-v3-turbo engine.
    
    Utilizes dynamic runtime API key rotation and fair-use load-balancing via SupabaseKeyManager.
    """
    start_time = time.time()
    timestamp_str = datetime.now(timezone.utc).isoformat()
    request_id = str(uuid.uuid4())

    # Extract file size tracking metrics safely
    try:
        file_object.seek(0, os.SEEK_END)
        file_size = file_object.tell()
        file_object.seek(0)  # Always rewind byte pointer stream indexes safely
    except Exception:
        file_size = 0

    # Primary Structural Integrity Validation Pass
    validation = validate_audio(filename, file_size, mime_type)
    if not validation["valid"]:
        return {
            "request_id": request_id,
            "success": False,
            "transcription": "",
            "detected_language": None,
            "provider": "Groq",
            "model": MODEL_NAME,
            "processing_time": 0.0,
            "timestamp": timestamp_str,
            "message": validation["message"]
        }

    transcription_text = ""
    normalized_lang = "Unknown"
    error_message = None
    success = False
    attempt = 0

    # Infinite retry loop matching llm_service pattern until success or out of keys
    while key_manager.has_active_keys():
        client = key_manager.get_groq_client()
        if client is None:
            error_message = "No operational Groq clients could be fetched from rotation manager."
            break

        # Dynamic structural tracking metrics tracking sequential failures
        attempt += 1
        current_key = key_manager.current_key_id or "Unknown ID"
        logger.info(f"[Req: {request_id}] Attempt {attempt} | Key ID: {current_key}")

        try:
            # Ensure the pointer is correctly at the absolute beginning of stream buffer for retries
            file_object.seek(0)
            
            response = client.audio.transcriptions.create(
                model=MODEL_NAME,
                file=(filename, file_object),
                response_format="json"
            )
            
            if hasattr(response, "text"):
                transcription_text = response.text
            elif isinstance(response, dict):
                transcription_text = response.get("text", "")
                
            if not transcription_text or not transcription_text.strip():
                # Logging check for blank/empty audio structures to optimize observability
                logger.warning(f"[Req: {request_id}] Audio contained no recognizable speech.")
                return {
                    "request_id": request_id,
                    "success": False,
                    "transcription": "",
                    "detected_language": None,
                    "provider": "Groq",
                    "model": MODEL_NAME,
                    "processing_time": round(time.time() - start_time, 3),
                    "timestamp": timestamp_str,
                    "message": "No speech detected."
                }
                
            raw_lang = "en"
            if hasattr(response, "language"):
                raw_lang = response.language
            elif isinstance(response, dict):
                raw_lang = response.get("language", "en")
                
            normalized_lang = LANGUAGE_MAP.get(raw_lang.lower(), "English")
            
            key_manager.mark_current_key_used()
            success = True
            
            logger.info(f"[Req: {request_id}] Whisper transcription completed successfully.")
            break  
            
        except (groq.AuthenticationError, groq.RateLimitError) as e:
            logger.warning(f"[Req: {request_id}] Key identity rejected/exhausted on Whisper engine. Dropping key token matrix.")
            key_manager.handle_quota_exhausted()
            error_message = f"Provider Matrix Exhausted: {str(e)}"
            continue  
            
        except (groq.APIConnectionError, TimeoutError, ConnectionError, OSError) as e:
            logger.warning(f"[Req: {request_id}] Transport infrastructure dropout occurred on current path. Cycling key matrix.")
            key_manager.cycle_key()
            error_message = f"Network Connection Dropout: {str(e)}"
            continue  

        except groq.APIStatusError as e:
            status_code = e.status_code
            if status_code in (401, 403, 429):
                logger.warning(f"[Req: {request_id}] Whisper key flagged with HTTP {status_code}. Executing key quota eviction.")
                key_manager.handle_quota_exhausted()
                error_message = f"APIStatusError ({status_code}): Quota eviction triggered."
                continue
            elif status_code in (500, 502, 503, 504):
                logger.warning(f"[Req: {request_id}] Whisper upstream down with HTTP {status_code}. Cycling to next available key.")
                key_manager.cycle_key()
                error_message = f"APIStatusError ({status_code}): Upstream transport retry triggered."
                continue
            else:
                error_message = f"APIStatusError: Server rejected configuration (Status: {status_code})."
                break

        except Exception as e:
            logger.exception(f"[Req: {request_id}] Unexpected Whisper runtime failure.")
            error_message = f"UnexpectedError: Whisper pipeline process crash: {type(e).__name__} - {str(e)}"
            break

    elapsed_time = round(time.time() - start_time, 3)

    # Performance Metric Logging
    ext = filename.split(".")[-1].lower() if "." in filename else "unknown"
    logger.info(
        f"[Req: {request_id}] Format: {ext} | Size: {round(file_size / 1024, 2)} KB | "
        f"Latency: {elapsed_time}s | Success: {success} | Lang: {normalized_lang} | Error: {error_message}"
    )

    return {
        "request_id": request_id,
        "success": success,
        "transcription": transcription_text.strip(),
        "detected_language": normalized_lang if success else None,
        "provider": "Groq",
        "model": MODEL_NAME,
        "processing_time": elapsed_time,
        "timestamp": timestamp_str,
        "message": "Transcription compiled successfully." if success else error_message
    }


def health_check() -> Dict[str, Any]:
    """Returns runtime connectivity status check parameters consumable by FastAPI lifecycles."""
    start_time = time.time()
    
    has_keys = key_manager.has_active_keys()
    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "healthy" if has_keys else "unhealthy",
        "provider": "Groq",
        "model": MODEL_NAME,
        "connection_status": has_keys,
        "latency_ms": elapsed_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "Whisper engine operational with hot pool key tracking." if has_keys else "No active Groq key layers available."
    }
