import os
import time  # 1. FIXED: Added explicitly to support health check metrics
import logging
import threading
from typing import Dict, Any, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection  # FIXED: Modernized import path
from chromadb.utils import embedding_functions

# =====================================================================
# Centralized Production Constants
# =====================================================================
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "chroma_db_storage")
EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

COLLECTION_METADATA_BLUEPRINT: Dict[str, Any] = {
    "hnsw:space": "cosine",  # Pinning standard distance metrics explicitly
    "project": "Appna Bank AI",
    "system_version": "1.0.0",
    "description": "Multilingual financial knowledge base vector storage partitions."
}

# Configure Production Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================================================================
# System Lifecycle Singletons & Cache Matrices
# =====================================================================
# 4. Thread-safe initialization primitives
_init_lock = threading.Lock()
_initialized: bool = False

# 2 & 3. Singleton runtime handles (Client and Embedding Model)
_chroma_client: Optional[ClientAPI] = None
_embedding_function: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None

# 5. Memory Cache map to eliminate repetitive lookup overhead
_collection_cache: Dict[str, Collection] = {}  # FIXED: Updated type hint to Collection


def initialize_database() -> None:
    """4. Thread-safe initialization routine using a double-check locking pattern.
    
    3. Loads the multilingual embedding model exactly once during startup 
    and keeps the application scalable under concurrent request contexts.
    """
    global _chroma_client, _embedding_function, _initialized
    
    if _initialized:
        return

    with _init_lock:
        # Double-check locking pattern block
        if _initialized:
            return
            
        try:
            # 8. Log Initialization Parameters Safely
            logging.info(f"Initializing Production ChromaDB engine at storage path: {CHROMA_DB_PATH}")
            
            # 7. Ensure directory workspace paths exist for persistent operations
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            # 2. Reusable client initialization
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            # 3. Explicit Embedding instantiation
            logging.info(f"Loading explicit Multilingual Embedding Engine: {EMBEDDING_MODEL_NAME}")
            _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
            
            _initialized = True
            logging.info("ChromaDB Core Layer and Embedding matrix handles established successfully.")
            
        except Exception as e:
            logging.critical(f"9. Fatal core error initializing database vector infrastructure: {str(e)}")
            raise RuntimeError(f"Database Core Matrix Initialization Failure: {str(e)}")


def get_collection(name: str) -> Collection:  # FIXED: Return type hint to Collection
    """Creates or fetches a cached collection instance partition.

    10. Separation of Concerns: Completely independent from downstream prompting architectures,
    answer builders, LLM providers, or network web tools.

    Args:
        name (str): Unique target structural partition lookup index handle.

    Returns:
        Collection: Cached reusable instance mapping query transactional utilities.
    """
    if not _initialized:
        initialize_database()
        
    # 5. Cache interception
    if name in _collection_cache:
        return _collection_cache[name]
        
    with _init_lock:
        if name in _collection_cache:
            return _collection_cache[name]
            
        try:
            logging.info(f"Cache miss on partition handle [{name}]. Accessing storage directory.")
            
            custom_metadata = COLLECTION_METADATA_BLUEPRINT.copy()
            custom_metadata["collection_identifier_name"] = name
            
            collection_instance = _chroma_client.get_or_create_collection(
                name=name,
                embedding_function=_embedding_function,
                metadata=custom_metadata
            )
            
            # 5. Write back to lookup collection cache map
            _collection_cache[name] = collection_instance
            logging.info(f"Collection partition instance context [{name}] successfully mounted inside memory cache.")
            
            return collection_instance
            
        except Exception as e:
            logging.error(f"9. Failed loading or creating collection reference identifier [{name}]: {str(e)}")
            raise ValueError(f"Collection initialization error exception failure on context name '{name}': {str(e)}")


def check_database() -> Dict[str, Any]:
    """6. Verifies the connectivity, availability, and telemetry status of database interfaces.
    
    Provides explicit telemetry parameters readily consumable by any external health endpoint monitor.

    Returns:
        Dict[str, Any]: Diagnostic metrics map charting runtime statistics.
    """
    start_time = time.time()
    is_healthy = False
    diagnostic_message = "Database components operational."
    cached_partitions_count = 0
    
    try:
        if not _initialized:
            initialize_database()
            
        if _chroma_client is None or _embedding_function is None:
            raise RuntimeError("Database singletons or model parameters are uninitialized.")
            
        # Issue an instant diagnostic query ping pulse to verify availability
        _chroma_client.heartbeat()
        cached_partitions_count = len(_collection_cache)
        is_healthy = True
        
    except Exception as e:
        diagnostic_message = f"Health check failed: {str(e)}"
        logging.error(f"9. Health status check failure exception log trace mapping: {diagnostic_message}")
        
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    # 6. Comprehensive Monitoring Metrics Mapping Output
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "database_reachable": is_healthy,
        "database_path": CHROMA_DB_PATH,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "cached_collections_count": cached_partitions_count,
        "latency_ms": elapsed_ms,
        "message": diagnostic_message
    }


# Automatically execute core structural configurations on startup
initialize_database()
