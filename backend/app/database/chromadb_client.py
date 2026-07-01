# backend/app/database/chromadb_client.py
import os
import time
import logging
import threading
import json
import socket
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

# =====================================================================
# Centralized Production Constants
# =====================================================================
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "chroma_db_storage")

# Dynamically load model identifier from environment context with safety fallback
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

COLLECTION_METADATA_BLUEPRINT: Dict[str, Any] = {
    "hnsw:space": "cosine",  # Enforcing strict cosine similarity
    "project": "Appna Bank AI",
    "system_version": "1.0.0",
    "description": "Multilingual financial knowledge base vector storage partitions."
}

# Improved Logging configuration to match rules (no basicConfig() inside this file)
logger = logging.getLogger(__name__)

# =====================================================================
# Custom Zero-RAM Serverless Embedding Function with Auto-Retry & Timeout
# =====================================================================
class ZeroRamHuggingFaceEmbedding(EmbeddingFunction):
    """Custom embedding function that routes text formatting tasks remotely
    to the current Hugging Face Inference API. Consumes 0MB local server memory.
    """
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise RuntimeError(
                "Critical Configuration Error: 'HUGGINGFACE_API_KEY' environment variable is missing. "
                "Remote embedding computations cannot proceed without a valid token."
            )
        self.api_key = api_key
        self.model_name = model_name.strip()
        
        # Route traffic through the current stable Hugging Face inference router
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        # Normalize incoming inputs into standard list strings
        texts = [input] if isinstance(input, str) else list(input)
        batch_size = len(texts)
        
        # Fixed: Removed the "options" configuration block entirely to stabilize payload structure
        payload = json.dumps({"inputs": texts}).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "AppnaBankAI/1.0"
        }

        # Parameters meeting requirement criteria precisely
        max_retries = 3
        backoff_delays = [1.0, 2.0, 4.0]  # Exponential backoff array sequence
        
        # Fixed: Extended timeout to 120 seconds to completely guard against upstream cold-starts
        timeout_seconds = 120.0

        logger.info(f"embedding request start: Dispatching remote vectors to Hugging Face API for batch size: {batch_size}")

        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(self.api_url, data=payload, headers=headers, method="POST")
                
                with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                    status_code = response.getcode()
                    logger.info(f"Hugging Face Remote API Status Code: {status_code} on attempt {attempt}")
                    
                    result = json.loads(response.read().decode("utf-8"))
                    
                    # Intercept dictionary error blocks early to expose descriptive upstream exceptions
                    if isinstance(result, dict):
                        logger.error(f"Hugging Face Dictionary Response Exception payload: {result}")
                        raise RuntimeError(result.get("error", "Unknown Hugging Face internal engine error encountered."))
                    
                    if not isinstance(result, list):
                        raise ValueError("Received an invalid response structure payload from Hugging Face endpoint.")
                    
                    logger.info(f"Embedding request successfully generated for batch size: {batch_size}")
                    return result

            except urllib.error.HTTPError as http_err:
                status_code = http_err.code
                error_body = ""
                try:
                    error_body = http_err.read().decode("utf-8")
                except Exception:
                    pass
                
                # Logs both the HTTP status code and response body when Hugging Face returns an error
                logger.error(f"HTTP Error {status_code} on attempt {attempt}: {http_err.reason} | Response Body: {error_body}")
                
                # Explicit handling for rate limits to optimize observability
                if status_code == 429:
                    logger.warning("Hugging Face rate limit reached.")

                # Immediately raise permanent errors. Do not retry client bugs or invalid schemas.
                if status_code in [400, 401, 403, 404, 422]:
                    logger.error(f"Permanent API Exception (HTTP {status_code}): {http_err.reason} - Failing immediately without retry.")
                    raise RuntimeError(f"Permanent API Exception (HTTP {status_code}): {http_err.reason} - {error_body}")
                
                # Transient errors updates retry paths
                if attempt == max_retries:
                    logger.exception(f"Embedding generation failed after {max_retries} attempts.")
                    raise RuntimeError(f"Failed to generate embeddings after {max_retries} attempts. HTTP Error: {status_code}")
                
            except (urllib.error.URLError, TimeoutError, socket.gaierror) as network_err:
                logger.exception(f"Transient network error/timeout/DNS anomaly on attempt {attempt}: {str(network_err)}")
                if attempt == max_retries:
                    logger.exception(f"Embedding generation failed after {max_retries} attempts.")
                    raise RuntimeError(f"Failed to generate embeddings due to network failure or timeout after {max_retries} attempts.")
                    
            except Exception as e:
                # Catch-all unexpected anomalies (e.g., parsing failures) - immediate fail to keep ingestion pipeline safe
                logger.exception(f"Non-transient or parsing error exception on attempt {attempt}: {str(e)}")
                raise RuntimeError(f"Unexpected processing exception inside embedding wrapper: {str(e)}")

            # Wait using explicit exponential backoff scale (1s, 2s, 4s) with enhanced logging
            current_delay = backoff_delays[attempt - 1]
            logger.warning(
                f"Embedding request failed.\n\n"
                f"Attempt:\n"
                f"{attempt} / {max_retries}\n\n"
                f"Retrying in\n"
                f"{int(current_delay)} seconds..."
            )
            time.sleep(current_delay)

        logger.exception(f"Embedding generation failed after {max_retries} attempts.")
        raise RuntimeError("Final failure: Embedding function evaluation route exited retry loop unexpectedly.")

# =====================================================================
# System Lifecycle Singletons & Cache Matrices
# =====================================================================
_init_lock = threading.Lock()
_initialized: bool = False

_chroma_client: Optional[ClientAPI] = None
_embedding_function: Optional[ZeroRamHuggingFaceEmbedding] = None
_collection_cache: Dict[str, Collection] = {}


def initialize_database() -> None:
    """Thread-safe instantiation sequence using a double-check locking layout.
    
    CRITICAL: Does not run automatically on import. Must be explicitly invoked
    from within the FastAPI application lifespan workflow inside main.py.
    """
    global _chroma_client, _embedding_function, _initialized
    
    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return
            
        try:
            start_init_time = time.time()
            logger.info(f"Initializing Production ChromaDB Persistent Architecture at: {CHROMA_DB_PATH}")
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            # Fetch authorization token securely
            hf_token = os.getenv("HUGGINGFACE_API_KEY", "")
            _embedding_function = ZeroRamHuggingFaceEmbedding(api_key=hf_token, model_name=EMBEDDING_MODEL_NAME)
            
            # Postpone validation checklist processing until first transactional access point
            logger.info("Remote embedding engine validation deferred. Health verification checks will occur lazily on first embedding runtime invocation context.")

            _initialized = True
            total_startup_time = round(time.time() - start_init_time, 4)
            
            logger.info(
                f"""
===================================

Appna Bank AI ChromaDB Ready

Database:
{CHROMA_DB_PATH}

Embedding Model:
{EMBEDDING_MODEL_NAME}

Endpoint:
{_embedding_function.api_url}

Startup Time:
{total_startup_time} seconds

Status:
READY

===================================
"""
            )
            
        except Exception as e:
            logger.critical(f"Fail-Fast Core Lockout: Database Vector Matrix failed initialization: {str(e)}")
            raise RuntimeError(f"Database Core Matrix Initialization Failure: {str(e)}")


def get_collection(name: str) -> Collection:
    """Creates or fetches a cached collection instance partition.
    Enforces safe execution under concurrent threaded requests.
    """
    if not _initialized:
        raise RuntimeError("Core Database is uninitialized. Ensure initialize_database() is fired on lifespan start.")
    
    if name in _collection_cache:
        return _collection_cache[name]
        
    with _init_lock:
        if name in _collection_cache:
            return _collection_cache[name]
            
        try:
            logger.info(f"Cache lookup miss for collection partition [{name}]. Initializing structural sync.")
            
            custom_metadata = COLLECTION_METADATA_BLUEPRINT.copy()
            custom_metadata["collection_identifier_name"] = name
            
            collection_instance = _chroma_client.get_or_create_collection(
                name=name,
                embedding_function=_embedding_function,
                metadata=custom_metadata
            )
            
            _collection_cache[name] = collection_instance
            logger.info(f"Collection partition instance [{name}] successfully written back to memory cache.")
            
            return collection_instance
            
        except Exception as e:
            logger.exception(f"Failed mounting or creating collection reference identifier [{name}]: {str(e)}")
            raise ValueError(f"Collection initialization error exception failure on context name '{name}': {str(e)}")


def check_database() -> Dict[str, Any]:
    """Verifies the health, network availability, and connection telemetry statistics of vector components.
    
    Optimized: Executes locally isolated fast diagnostics exclusively. Network dependencies are skipped 
    to protect the health endpoint response times from external downtime interference patterns.
    """
    start_time = time.time()
    is_healthy = False
    diagnostic_message = "All database modules responding optimally."
    cached_partitions_count = 0
    
    try:
        if not _initialized:
            raise RuntimeError("Database component layer is currently uninitialized.")
            
        if _chroma_client is None or _embedding_function is None:
            raise RuntimeError("Database configuration runtime matrices are broken or uninstantiated.")
            
        # Execute rapid diagnostic local heartbeat pulse check exclusively
        _chroma_client.heartbeat()
        
        cached_partitions_count = len(_collection_cache)
        is_healthy = True
        
    except Exception as e:
        diagnostic_message = f"Health diagnostic check failed: {str(e)}"
        logger.exception(f"Health status assessment failure exception tracking log: {diagnostic_message}")
        
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "database_reachable": is_healthy,
        "database_path": CHROMA_DB_PATH,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cached_collections_count": cached_partitions_count,
        "latency_ms": elapsed_ms,
        "message": diagnostic_message
    }
