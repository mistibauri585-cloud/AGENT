# backend/app/database/chromadb_client.py
import os
import time
import logging
import threading
from typing import Dict, Any, Optional, List
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from huggingface_hub import InferenceClient

# =====================================================================
# Centralized Production Constants
# =====================================================================
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "chroma_db_storage")

# Read the embedding model from environment variables with the requested default
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "BAAI/bge-m3"
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
# Official SDK HuggingFace Embedding Function with Thread Safety
# =====================================================================
class ZeroRamHuggingFaceEmbedding(EmbeddingFunction):
    """Custom embedding function that routes text embedding tasks remotely
    via the official huggingface_hub InferenceClient. Consumes 0MB local server memory.
    """
    def __init__(self, api_key: Optional[str], model_name: str):
        if not api_key:
            logger.critical("Initialization failed: 'HUGGINGFACE_API_KEY' environment variable is missing.")
            raise RuntimeError(
                "Critical Configuration Error: 'HUGGINGFACE_API_KEY' environment variable is missing. "
                "Remote embedding computations cannot proceed without a valid token."
            )
        
        self.model_name = model_name.strip()
        
        # Instantiate the official SDK client using the serverless hf-inference provider layout
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )
        logger.info(f"Hugging Face InferenceClient initialized successfully using model: {self.model_name}")

    def __call__(self, input: Documents) -> Embeddings:
        # Normalize incoming inputs into standard list strings
        texts = [input] if isinstance(input, str) else list(input)
        batch_size = len(texts)
        
        logger.info(f"Embedding request start: Dispatching {batch_size} text inputs to huggingface_hub InferenceClient.")
        
        try:
            # Fixed: Explicitly mapping to the text keyword argument for modern SDK syntax alignment
            response = self.client.feature_extraction(
                text=texts,
                model=self.model_name
            )
            
            # Fail gracefully and defensively if feature_extraction returns an invalid structure or None
            if response is None:
                raise ValueError("InferenceClient returned an empty response payload (None).")
            
            # Handle NumPy arrays or different return sequences gracefully if converted by SDK wrapper layers
            if hasattr(response, "tolist"):
                embeddings = response.tolist()
            else:
                embeddings = response

            if not isinstance(embeddings, list):
                raise ValueError(f"Invalid response structure type: Expected list but got {type(embeddings).__name__}")
            
            logger.info(f"Embedding vectors successfully compiled via huggingface_hub for batch size: {batch_size}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to extract features via huggingface_hub SDK: {str(e)}", exc_info=True)
            raise RuntimeError(f"Hugging Face Inference SDK Exception tracking point: {str(e)}")

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
            
            _initialized = True
            total_startup_time = round(time.time() - start_init_time, 4)
            
            logger.info(
                f"""
===================================
Appna Bank AI ChromaDB Ready
Database Path:    {CHROMA_DB_PATH}
Embedding Model:  {EMBEDDING_MODEL_NAME}
Startup Time:     {total_startup_time} seconds
Status:           READY
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
