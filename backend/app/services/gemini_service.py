import os
import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List
from groq import Groq
import groq
from langdetect import detect, DetectorFactory

# Set seed for reproducible local language detection results
DetectorFactory.seed = 0

# Configure production logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [Req: %(pathname)s] - %(message)s")

# =====================================================================
# 3. MODEL CONFIGURATION CONSTANTS
# =====================================================================
MODEL_NAME = "qwen/qwen3-32b"
TEMPERATURE = 0.3
MAX_TOKENS = 1000
TOP_P = 0.9
STREAM_MODE = False  # 11. Readily toggled for future streaming pipelines

# 2. CONTEXT LIMITS
MAX_CONTEXT_CHUNKS = 5
MAX_CONTEXT_CHAR_LIMIT = 8000 

# =====================================================================
# 4. EXPANDED LANGUAGE MAPPING
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

# =====================================================================
# 1 & 10. SINGLETON CLIENT INITIALIZATION (PREPARED FOR API MANAGER)
# =====================================================================
def _initialize_groq_client() -> Groq:
    """Initializes the Groq client once at application startup.
    
    Can be seamlessly redirected to pull keys dynamically from an API Key Manager
    without disrupting downstream business logic.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logging.warning("GROQ_API_KEY environment variable is missing at lifecycle initiation.")
    return Groq(api_key=api_key)

# Instantiated once on initialization module-level load
groq_client = _initialize_groq_client()


def _detect_language_locally(text: str) -> str:
    """Detects the ISO 639-1 language code locally.
    
    Falls back to 'English' if the language is unknown or detection fails.
    """
    try:
        lang_code = detect(text)
        return LANGUAGE_MAP.get(lang_code, "English")
    except Exception:
        return "English"


def _process_context(raw_context: Any) -> str:
    """2. Validates, limits, and truncates retrieved context chunks to preserve 
    token bandwidth and improve RAG performance accuracy.
    """
    if not raw_context:
        return ""
        
    # If context arrives as a list of chunks, combine up to the specified maximum
    if isinstance(raw_context, list):
        target_chunks = raw_context[:MAX_CONTEXT_CHUNKS]
        combined = "\n\n".join([str(chunk).strip() for chunk in target_chunks])
    else:
        combined = str(raw_context).strip()
        
    # Apply global hard character truncation threshold
    return combined[:MAX_CONTEXT_CHAR_LIMIT]


# =====================================================================
# CORE LLM SERVICE INTERFACE
# =====================================================================
def ask_the_principal(user_query: str, retrieved_context: Any) -> Dict[str, Any]:
    """Processes an incoming user financial query via RAG using the Groq API.

    14. Future Ready Design: Dedicated strictly to Language Routing, Prompt Building, 
    API execution, and response structural formatting.

    Args:
        user_query (str): The raw text message input provided by the end user.
        retrieved_context (Any): Raw context string or list of text chunks fetched 
                                 from the vector knowledge base.

    Returns:
        Dict[str, Any]: Structured operational payload containing the generated response, 
                        execution diagnostics, and pipeline tracking metadata metrics.
    """
    start_time = time.time()
    
    # 7. Generate a unique tracking UUID for traceability mapping across services
    request_id = str(uuid.uuid4())
    timestamp_str = datetime.now(timezone.utc).isoformat()
    
    # 2 & 4. Run performance extraction layers locally
    detected_language = _detect_language_locally(user_query)
    clean_context = _process_context(retrieved_context)
    
    # 13. Context Validation Boundary
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

    # 5. Formulate strict, non-duplicate production-hardened instructions
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

    # 6. Streamlined clear block structure (No redundant language rules inside user block)
    full_prompt = f"""
========== KNOWLEDGE BASE ==========
{knowledge_block}

========== USER QUESTION ==========
{user_query}
"""

    try:
        # 1. Reuses the application startup client directly
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
        success = True
        error_type = None

    # 6. Comprehensive API Error Routing Boundaries
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

    # 9. Clean audit logging (Excludes internal queries, system prompts, or confidential customer parameters)
    logging.info(
        f"Req ID: {request_id} | Model: {MODEL_NAME} | Lang: {detected_language} | "
        f"Latency: {elapsed_time}s | Success: {success} | Error Type: {error_type}"
    )

    # 8. Complete analytic data payload
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
