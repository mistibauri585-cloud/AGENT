import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import groq
from langdetect import detect, DetectorFactory
# 5. Clean Architectural Import: Relies on api_key_manager for client state
from app.services.api_key_manager import _get_groq_client

# Set seed for reproducible local language detection results
DetectorFactory.seed = 0

# 8. Production Logging Configuration (Tracks metadata without bleeding sensitive text context)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [Req: %(pathname)s] - %(message)s")

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
# 2. SUPPORTED MULTILINGUAL LOCAL MATRIX
# =====================================================================
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


def _detect_language_locally(text: str) -> str:
    """2. Analyzes user strings locally on CPU using ISO maps to save API costs.
    
    Defaults cleanly to English if detection bounds fail.
    """
    try:
        lang_code = detect(text)
        return LANGUAGE_MAP.get(lang_code, "English")
    except Exception:
        return "English"


def _process_context(raw_context: Any) -> str:
    """3. Validates incoming chunks, protects token bounds, and handles truncations 
    to insulate Groq against token inflation.
    """
    if not raw_context:
        return ""
        
    if isinstance(raw_context, list):
        # Filter empty lines or falsy values from structural payloads
        valid_chunks = [str(chunk).strip() for chunk in raw_context if chunk and str(chunk).strip()]
        target_slices = valid_chunks[:MAX_CONTEXT_CHUNKS]
        combined = "\n\n".join(target_slices)
    else:
        combined = str(raw_context).strip()
        
    # Apply hard maximum character limits safely
    return combined[:MAX_CONTEXT_CHAR_LIMIT]


# =====================================================================
# 1. CORE LLM GENERATION WORKSPACE (SINGLE RESPONSIBILITY ENGINE)
# =====================================================================
def ask_the_principal(user_query: str, retrieved_context: Any) -> Dict[str, Any]:
    """6. Formulates system-instructions and routes operational prompts over Groq client bindings.
    
    Strict Design Boundary: Has no exposure to core DB queries, Whisper audio strings, 
    Redis, or client credentials. Expects clean runtime strings.
    """
    start_time = time.time()
    
    # 9. Traceability tracking properties instantiation
    request_id = str(uuid.uuid4())
    timestamp_str = datetime.now(timezone.utc).isoformat()
    
    # Executing operational preparation sub-layers
    detected_language = _detect_language_locally(user_query)
    clean_context = _process_context(retrieved_context)
    
    # Context Validation Fallback Configuration Block
    if not clean_context:
        context_instruction = (
            "CRITICAL: No reliable information or localized database matches were found for this query.\n"
            "Politely inform the user that reliable local information is not available at the moment, "
            "and suggest alternative official channels if appropriate. Do not attempt to guess or fulfill requirements."
        )
        knowledge_block = "[NO LOCAL KNOWLEDGE BASE AVAILABLE]"
    else:
        context_instruction = (
            "Use the provided context block first to construct your answer. "
            "If the answer cannot be confidently verified using this context, explicitly tell the user "
            "that reliable local information is not available. Do not generate unverified details."
        )
        knowledge_block = clean_context

    # 4. Prompt Engineering System Instruction Blueprint
    system_instruction = (
        "You are 'Appna Bank AI', a warm financial companion for farmers, students, and rural communities.\n\n"
        "RESPONSE RULES:\n"
        f"1. You must reply completely in {detected_language}.\n"
        "2. Explain everything like a teacher talking to a Class 5 student. Use simple analogies and zero complex jargon.\n"
        f"3. {context_instruction}\n"
        "4. Strict Hallucination Protection: Do not invent facts, financial rates, rules, or claims. Trustworthiness is critical.\n\n"
        "FINANCIAL SAFETY MANDATES:\n"
        "- Prioritize and explicitly recommend building a robust Emergency Fund before allocating money elsewhere.\n"
        "- Encourage securing basic Insurance protection frameworks prior to making any volatile investments.\n"
        "- Direct users to applicable Indian Government development schemes (e.g., PM-Kisan, Sukanya Samriddhi) where logical.\n"
        "- Absolute restriction: Never dispense risky stock updates or speculative advice to seniors or low-income households."
    )

    full_prompt = f"""
========== KNOWLEDGE BASE ==========
{knowledge_block}

========== USER QUESTION ==========
{user_query}
"""

    success = True
    error_type = None
    answer = ""

    # 5. Delegate Client Construction Lifecycle to separate service
    groq_client = _get_groq_client()
    if not groq_client:
        return {
            "request_id": request_id,
            "timestamp": timestamp_str,
            "detected_language": detected_language,
            "answer": "System configuration error: No active keys found in the backend database mapping clusters.",
            "model": MODEL_NAME,
            "success": False,
            "source": "Groq Key Error Routing Service Node",
            "response_time": round(time.time() - start_time, 3)
        }

    try:
        # 6. Execute core operational transaction payload over Groq client wrapper
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

    # 7. Comprehensive Error Routing Context Mapping (Returns user-friendly alerts)
    except groq.AuthenticationError:
        success = False
        error_type = "AuthenticationError"
        answer = "I am currently facing authentication issues logging into my backend network. Please check back soon."
        
    except groq.RateLimitError:
        success = False
        error_type = "RateLimitError"
        answer = "Our servers are experiencing heavy traffic. Please wait a minute and try asking again."
        
    except groq.APIStatusError as e:
        success = False
        error_type = f"APIStatusError({e.status_code})"
        answer = "My processing systems are taking a quick break. Let's try again in a few moments."
        
    except groq.APIConnectionError:
        success = False
        error_type = "APIConnectionError"
        answer = "I am having trouble connecting to the network layer. Please verify your connection health."
        
    except Exception as e:
        success = False
        error_type = f"UnexpectedError: {type(e).__name__}"
        answer = "An unexpected error occurred while compiling your financial response guidance."

    elapsed_time = round(time.time() - start_time, 3)

    # 8. Clean Audit Metric Tracking Log Call (Excludes prompts and customer queries)
    logging.info(
        f"Req ID: {request_id} | Model: {MODEL_NAME} | Lang: {detected_language} | "
        f"Latency: {elapsed_time}s | Success: {success} | Error Type: {error_type}"
    )

    # 9. Standardized Structural Analytics Response Payload Object Frame Output
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
