import os
import time
import uuid  # 1. Added for cross-service traceability mapping
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, BinaryIO
from groq import Groq
import groq

# Configure production logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [Whisper] - %(message)s")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
WHISPER_API_KEY: str = os.getenv("GROQ_WHISPER_API_KEY", "")
MODEL_NAME: str = "whisper-large-v3-turbo"

# Content validation constraints
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "webm", "ogg", "flac"}
# 3. Explicit production MIME type whitelist arrays
ALLOWED_MIME_TYPES = {"audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg", "audio/flac", "audio/x-wav"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB Hard ceiling cap boundary

# 2. Standardized Language Mapping matrix to align fully with llm_service.py
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu"
}

# Thread-safe initialization primitive
_init_lock = threading.Lock()

# Singleton Runtime Handles
_whisper_client: Optional[Groq] = None


def connect() -> None:
    """5. Initialized from FastAPI startup lifespan event loops.
    
    Thread-safe instantiation utilizing double-check locking patterns to ensure 
    only one client engine instance is mapped per application cluster layer.
    """
    global _whisper_client
    
    if _whisper_client is not None:
        return

    with _init_lock:
        if _whisper_client is not None:
            return
            
        try:
            if not WHISPER_API_KEY:
                logging.warning("GROQ_WHISPER_API_KEY environment variable is not configured.")
            
            logging.info("Initializing Singleton Groq Whisper Client Engine...")
            _whisper_client = Groq(api_key=WHISPER_API_KEY)
            logging.info("Groq Whisper Engine client initialized successfully.")
        except Exception as e:
            logging.critical(f"Fatal crash setting up runtime Whisper client singleton state: {str(e)}")
            _whisper_client = None


def _get_client() -> Optional[Groq]:
    """Internal runtime reference mapping accessor hook handle."""
    return _whisper_client


def validate_audio(filename: str, file_size_bytes: int, mime_type: Optional[str] = None) -> Dict[str, Any]:
    """Public helper implementation to audit structural payload integrity bounds.
    
    Checks filename extensions, validates content sizes, and verifies MIME type constraints.
    """
    if not filename:
        return {"valid": False, "message": "Missing file identifier signature metadata."}
        
    # Check format extension parameters
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False, 
            "message": f"Unsupported audio data format type extension: [.{ext}]. Supported: {list(ALLOWED_EXTENSIONS)}"
        }
        
    # 3. Validate content MIME type headers explicitly if supplied by FastAPI
    if mime_type and mime_type.lower() not in ALLOWED_MIME_TYPES:
        return {
            "valid": False,
            "message": f"Unsupported audio content MIME type detected: [{mime_type}]. Rejected before API transfer."
        }
        
    # Check physical mass footprint thresholds
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

    Strict Separation of Concerns: Independent of downstream generation layers, prompts, 
    ChromaDB partitions, or Redis keys. Expects and processes audio binary payloads only.
    """
    start_time = time.time()
    timestamp_str = datetime.now(timezone.utc).isoformat()
    
    # 1. Generate a unique tracking UUID for traceability mapping across services
    request_id = str(uuid.uuid4())

    # Extract file size tracking metrics safely
    try:
        file_object.seek(0, os.SEEK_END)
        file_size = file_object.tell()
        file_object.seek(0)  # Always rewind byte pointer stream indexes safely
    except Exception:
        file_size = 0

    # Primary Structural Integrity Validation Pass (Includes MIME Checks)
    validation = validate_audio(filename, file_size, mime_type)
    if not validation["valid"]:
        return {
            "request_id": request_id,  # 1. Included in response structure
            "success": False,
            "transcription": "",
            "detected_language": None,
            "provider": "Groq",
            "model": MODEL_NAME,
            "processing_time": 0.0,
            "timestamp": timestamp_str,
            "message": validation["message"]
        }

    client = _get_client()
    if client is None:
        logging.error(f"[Req: {request_id}] Transcription dropped: Singleton client missing initialization parameters.")
        return {
            "request_id": request_id,
            "success": False,
            "transcription": "",
            "detected_language": None,
            "provider": "Groq",
            "model": MODEL_NAME,
            "processing_time": 0.0,
            "timestamp": timestamp_str,
            "message": "Whisper audio processing service context is currently offline."
        }

    success = False
    transcription_text = ""
    normalized_lang = "Unknown"
    error_message = None

    try:
        # Dispatch binary package parameters over Groq Whisper client mapping
        response = client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=(filename, file_object),
            response_format="json"
        )
        
        # Pull text attributes safely out of response formats
        if hasattr(response, "text"):
            transcription_text = response.text
        elif isinstance(response, dict):
            transcription_text = response.get("text", "")
            
        # 4. Check for clean structural empty audio/blank transcription responses
        if not transcription_text or not transcription_text.strip():
            logging.warning(f"[Req: {request_id}] Whisper returned an empty payload string. No vocal tokens generated.")
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
            
        # Extract and parse raw short language flags returned by the provider
        raw_lang = "en"
        if hasattr(response, "language"):
            raw_lang = response.language
        elif isinstance(response, dict):
            raw_lang = response.get("language", "en")
            
        # 2. Normalize raw Whisper output string codes down to target database names
        normalized_lang = LANGUAGE_MAP.get(raw_lang.lower(), "English")
        success = True
        
    # Comprehensive Exception Routing Boundaries
    except groq.AuthenticationError:
        error_message = "AuthenticationError: Groq client validation credential checks dropped."
    except groq.APIConnectionError:
        error_message = "APIConnectionError: Internal network path transmission timeout failure."
    except groq.APIStatusError as e:
        error_message = f"APIStatusError: Server rejected configuration instructions (Status Code: {e.status_code})."
    except Exception as e:
        error_message = f"UnexpectedError: Pipeline runtime collapse processing text sequence mapping: {type(e).__name__}"

    elapsed_time = round(time.time() - start_time, 3)

    # 1. Clean Performance Metric Logs containing the Request ID
    ext = filename.split(".")[-1].lower() if "." in filename else "unknown"
    logging.info(
        f"[Req: {request_id}] Format: {ext} | Size: {round(file_size / 1024, 2)} KB | "
        f"Latency: {elapsed_time}s | Success: {success} | Lang: {normalized_lang} | Error: {error_message}"
    )

    return {
        "request_id": request_id,  # 1. Standardized unique identity tracing field
        "success": success,
        "transcription": transcription_text.strip(),
        "detected_language": normalized_lang if success else None,  # 2. Verified Normalized Format
        "provider": "Groq",
        "model": MODEL_NAME,
        "processing_time": elapsed_time,
        "timestamp": timestamp_str,
        "message": error_message if error_message else "Transcription compiled successfully."
    }


def health_check() -> Dict[str, Any]:
    """Returns runtime connectivity status check parameters consumable by FastAPI lifecycles."""
    start_time = time.time()
    is_operational = False
    status_message = "Whisper audio processing pipelines configured."

    try:
        client = _get_client()
        if client is not None:
            if WHISPER_API_KEY:
                is_operational = True
            else:
                status_message = "Configuration key signature array mappings missing."
        else:
            status_message = "Core client interface context is currently uninitialized or offline."
    except Exception as e:
        status_message = f"Structural health probe exception trace event: {str(e)}"

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "healthy" if is_operational else "unhealthy",
        "provider": "Groq",
        "model": MODEL_NAME,
        "connection_status": is_operational,
        "latency_ms": elapsed_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": status_message
    }

# 5. Lifespan Ready: Implicit runtime initialization on load is completely disabled.
# Ensure you invoke `whisper_service.connect()` within your FastAPI lifespan startup framework.
