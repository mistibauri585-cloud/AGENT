import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import groq
from langdetect import detect, DetectorFactory

# FIXED: Import the actual manager class instead of the missing function
from app.services.api_key_manager import SupabaseKeyManager

# Set seed for reproducible local language detection results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Initialize the manager singleton instance globally at module load time
try:
    key_manager = SupabaseKeyManager()
except Exception as e:
    logger.critical(f"Failed to instantiate SupabaseKeyManager: {str(e)}")
    key_manager = None

# =====================================================================
# MODEL CONFIGURATION & CONTEXT CONSTANTS
# =====================================================================
MODEL_NAME = "qwen/qwen3-32b"
TEMPERATURE = 0.3
MAX_TOKENS = 1000
TOP_P = 0.9
STREAM_MODE = False

MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHAR_LIMIT = 8000 

# =====================================================================
# SUPPORTED MULTILINGUAL LOCAL MATRIX
# =====================================================================
LANGUAGE_MAP = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "mr": "Marathi",
    "ta": "Tamil", "te": "Telugu", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"
}

def _detect_language_locally(text: str) -> str:
    try:
        lang_code = detect(text)
        return LANGUAGE_MAP.get(lang_code, "English")
    except Exception:
        return "English"

def _process_context(raw_context: Any) -> str:
    if not raw_context:
        return ""
    if isinstance(raw_context, list):
        valid_chunks = [str(chunk).strip() for chunk in raw_context if chunk and str(chunk).strip()]
        target_slices = valid_chunks[:MAX_CONTEXT_CHUNKS]
        combined = "\n\n".join(target_slices)
    else:
        combined = str(raw_context).strip()
    return combined[:MAX_CONTEXT_CHAR_LIMIT]

# =====================================================================
# CORE LLM GENERATION WORKSPACE
# =====================================================================
def ask_the_principal(user_query: str, retrieved_context: Any) -> Dict[str, Any]:
    start_time = time.time()
    request_id = str(uuid.uuid4())
    timestamp_str = datetime.now(timezone.utc).isoformat()
    
    detected_language = _detect_language_locally(user_query)
    clean_context = _process_context(retrieved_context)
    
    if not clean_context:
        context_instruction = (
            "CRITICAL: No reliable information or localized database matches were found for this query.\n"
            "Politely inform the user that reliable local information is not available at the moment."
        )
        knowledge_block = "[NO LOCAL KNOWLEDGE BASE AVAILABLE]"
    else:
        context_instruction = (
            "Use the provided context block first to construct your answer. "
            "If the answer cannot be confidently verified using this context, explicitly tell the user "
            "that reliable local information is not available."
        )
        knowledge_block = clean_context

    system_instruction = (
        "You are 'Appna Bank AI', a warm financial companion for farmers, students, and rural communities.\n\n"
        "RESPONSE RULES:\n"
        f"1. You must reply completely in {detected_language}.\n"
        "2. Explain everything like a teacher talking to a Class 5 student.\n"
        f"3. {context_instruction}\n"
        "4. Strict Hallucination Protection: Do not invent facts.\n\n"
        "FINANCIAL SAFETY MANDATES:\n"
        "- Prioritize and explicitly recommend building a robust Emergency Fund.\n"
        "- Encourage securing basic Insurance protection frameworks.\n"
        "- Direct users to applicable Indian Government development schemes (e.g., PM-Kisan).\n"
        "- Never dispense risky stock updates or speculative advice."
    )

    full_prompt = f"========== KNOWLEDGE BASE ==========\n{knowledge_block}\n\n========== USER QUESTION ==========\n{user_query}"

    # FIXED: Resolve client using the manager instance wrapper method
    groq_client = None
    if key_manager:
        try:
            # Adjust method name if your class exposes it differently (e.g., key_manager.get_client())
            groq_client = key_manager.get_groq_client() 
        except Exception as e:
            logger.error(f"[{request_id}] Failed to acquire rotated Groq client: {str(e)}")

    if not groq_client:
        return {
            "request_id": request_id,
            "timestamp": timestamp_str,
            "detected_language": detected_language,
            "answer": "System configuration error: No active keys found in the rotation client pools.",
            "model": MODEL_NAME,
            "success": False,
            "source": "Groq Key Error Routing Service Node",
            "response_time": round(time.time() - start_time, 3)
        }

    success = True
    error_type = None
    answer = ""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            stream=STREAM_MODE
        )
        answer = response.choices[0].message.content

    except groq.AuthenticationError:
        success = False
        error_type = "AuthenticationError"
        answer = "I am currently facing authentication issues logging into my backend network."
        # If your manager has a mark_as_exhausted or error callback, trigger it here:
        # key_manager.mark_key_exhausted()
        
    except groq.RateLimitError:
        success = False
        error_type = "RateLimitError"
        answer = "Our servers are experiencing heavy traffic. Please wait a minute and try asking again."
        
    except Exception as e:
        success = False
        error_type = f"UnexpectedError: {type(e).__name__}"
        answer = "An unexpected error occurred while compiling your financial response guidance."

    elapsed_time = round(time.time() - start_time, 3)
    logger.info(f"Req ID: {request_id} | Model: {MODEL_NAME} | Latency: {elapsed_time}s | Success: {success}")

    return {
        "request_id": request_id,
        "timestamp": timestamp_str,
        "detected_language": detected_language,
        "answer": answer,
        "model": MODEL_NAME,
        "success": success,
        "source": "Groq",
        "response_time": elapsed_time
    }
