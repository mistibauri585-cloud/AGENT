# backend/app/services/llm_service.py
import time
import uuid
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import groq
from langdetect import detect, DetectorFactory

# Import the actual manager class handling the database state rotation
from app.services.api_key_manager import SupabaseKeyManager

# Set seed for reproducible local language detection results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# Initialize the manager singleton instance globally at module load time
try:
    key_manager = SupabaseKeyManager()
except Exception as e:
    logger.critical(f"Failed to instantiate SupabaseKeyManager: {str(e)}", exc_info=True)
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
            "Use the supplied knowledge context as the primary source of truth. "
            "If the answer cannot be found in the supplied context, politely tell the user that reliable information is unavailable. "
            "Do not invent facts."
        )
        knowledge_block = clean_context

    system_instruction = (
        "You are 'Appna Bank AI', a warm financial companion for farmers, students, and rural communities.\n\n"
        "STRICT ANSWER OUTPUT CONSTRAINT POLICY:\n"
        "- Return ONLY the final answer. Never expose your internal reasoning.\n"
        "- Never output <think> or </think> tags under any circumstances.\n"
        "- Never output markdown thinking blocks or ```thinking code blocks.\n"
        "- Never reveal your chain of thought, planning, or internal analysis to the user.\n"
        "- Never explain how the answer was generated or structured.\n"
        "- If you generate internal reasoning, discard it completely before sending the response.\n"
        "- Your final output must contain only the answer that the user should read.\n"
        "- If internal reasoning exists in your processing space, you must strip it completely before returning your final response.\n\n"
        "RESPONSE RULES:\n"
        f"1. You must reply completely in {detected_language}.\n"
        "2. Explain everything like a warm teacher talking to a Class 5 student.\n"
        f"3. {context_instruction}\n"
        "4. Strict Hallucination Protection: Do not invent facts.\n\n"
        "FINANCIAL SAFETY MANDATES:\n"
        "- Prioritize and explicitly recommend building a robust Emergency Fund.\n"
        "- Encourage securing basic Insurance protection frameworks.\n"
        "- Direct users to applicable Indian Government development schemes (e.g., PM-Kisan).\n"
        "- Never dispense risky stock updates or speculative advice."
    )

    full_prompt = f"========== KNOWLEDGE BASE ==========\n{knowledge_block}\n\n========== USER QUESTION ==========\n{user_query}"

    success = False
    answer = ""
    attempt = 0
    backoff_delays = [1.0, 2.0, 4.0]

    if not key_manager:
        logger.critical(f"[{request_id}] Key manager runtime dependency object initialization is broken.")
        return {
            "request_id": request_id,
            "timestamp": timestamp_str,
            "detected_language": detected_language,
            "answer": "No active Groq API keys are currently available.\n\nPlease try again later.",
            "model": MODEL_NAME,
            "success": False,
            "source": "Groq",
            "response_time": round(time.time() - start_time, 3)
        }

    # Infinite Loop Driven By Dynamic Database Presence Checks
    while key_manager.has_active_keys():
        attempt += 1
        attempt_start_time = time.time()
        
        # Load current Groq client from the dynamic database pool instance
        groq_client = key_manager.get_groq_client()
        current_key_id = key_manager.current_key_id or "unknown_key_id"

        if not groq_client:
            logger.warning(f"[{request_id}] [Attempt {attempt}] Failed to extract operational client reference. Re-verifying pool...")
            if not key_manager.has_active_keys():
                break
            continue

        try:
            logger.info(f"[{request_id}] [Attempt {attempt}] Forwarding completion transaction via Key ID: {current_key_id}")
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
            
            raw_content = response.choices[0].message.content or ""
            
            # Post-processing sanitization: Clean up XML tags or Markdown blocks detailing reasoning
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL | re.IGNORECASE)
            cleaned_content = re.sub(r'```(?:thinking|think|reasoning)?[\s\S]*?```', '', cleaned_content, flags=re.IGNORECASE)
            
            answer = cleaned_content.strip()
            answer = re.sub(r"\n{3,}", "\n\n", answer)
            
            if not answer:
                answer = (
                    "Sorry, I couldn't generate a valid response. "
                    "Please try again."
                )
                
            success = True
            
            # Rebalance queue priorities by pushing this successful key back and shifting downstream nodes forward
            key_manager.mark_current_key_used()
            
            latency = round(time.time() - attempt_start_time, 3)
            logger.info(
                f"[{request_id}] Attempt SUCCESS -> Attempt: {attempt} | Key ID: {current_key_id} | "
                f"Model: {MODEL_NAME} | Latency: {latency}s | Success: True | Error: None"
            )
            break  # Break loop instantly upon processing transaction successfully

        except (groq.AuthenticationError, groq.RateLimitError) as permanent_api_err:
            error_type = type(permanent_api_err).__name__
            latency = round(time.time() - attempt_start_time, 3)
            
            logger.warning(
                f"[{request_id}] Attempt FAILED -> Attempt: {attempt} | Key ID: {current_key_id} | "
                f"Model: {MODEL_NAME} | Latency: {latency}s | Success: False | Error: {error_type}"
            )
            
            # Instantly burn and cycle out the exhausted configuration token record from database availability pipelines
            key_manager.handle_quota_exhausted()

        except (TimeoutError, ConnectionError, OSError, groq.APIConnectionError) as transient_network_err:
            error_type = type(transient_network_err).__name__
            latency = round(time.time() - attempt_start_time, 3)
            
            logger.warning(
                f"[{request_id}] Attempt FAILED (Transient) -> Attempt: {attempt} | Key ID: {current_key_id} | "
                f"Model: {MODEL_NAME} | Latency: {latency}s | Success: False | Error: {error_type}"
            )
            
            # Cycle to the next key block configuration structure mapping without marking it dead
            key_manager.cycle_key()

        except Exception as e:
            error_type = f"UnexpectedError: {type(e).__name__}"
            latency = round(time.time() - attempt_start_time, 3)
            logger.exception(
                f"[{request_id}] Unhandled runtime error trace hit in processing engine structure pipeline on attempt {attempt}:"
            )
            raise RuntimeError(f"Unchecked exception during synthesis processing block pipeline: {str(e)}") from e

        # Apply exponential backing off logic rules if transactional sequence fails to settle context successfully
        if not success and key_manager.has_active_keys():
            # Pin max backup delay parameter safely at 4.0 seconds ceiling limit boundaries
            current_delay = backoff_delays[min(attempt - 1, len(backoff_delays) - 1)]
            logger.warning(f"[{request_id}] Attempt {attempt} failed. Retrying next connection profile node in {current_delay}s...")
            time.sleep(current_delay)

    # Exhaustive Pool Fallback Scenario Logging and Error Mask Assignment Mapping Flow
    if not success:
        answer = "No active Groq API keys are currently available.\n\nPlease try again later."
        logger.error(f"[{request_id}] Core Lockout Error: Infinite Key Loop terminated because remote pool records are depleted completely.")

    elapsed_time = round(time.time() - start_time, 3)
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
