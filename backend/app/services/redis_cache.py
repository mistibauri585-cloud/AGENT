# backend/app/services/redis_cache.py
import os
import time
import json
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import redis

# Configure clean, isolated module-scoped logger (No basicConfig overrides)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_TTL_SECONDS: int = int(os.getenv("REDIS_CACHE_TTL", "86400"))  # Default: 24 Hours
REDIS_SOCKET_TIMEOUT: float = 2.0  # Prevents blocking FastAPI workers if network drops

# Thread-safe initialization primitive
_init_lock = threading.Lock()

# Singleton Runtime References
_redis_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


def connect() -> None:
    """Initializes a global thread-safe Redis connection pool and client.
    
    Uses double-check locking. Proactively verifies connectivity via client.ping().
    If the ping fails, logs the error and leaves the client uninitialized.
    """
    global _redis_pool, _redis_client
    
    if _redis_client is not None:
        return

    with _init_lock:
        # Double-check locking pattern block
        if _redis_client is not None:
            return
            
        try:
            logger.info("Initializing Singleton Redis Connection Pool...")
            pool = redis.ConnectionPool.from_url(
                url=REDIS_URL,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                retry_on_timeout=True,
                decode_responses=True
            )
            client = redis.Redis(connection_pool=pool)
            
            # Proactively verify active connection signature
            logger.info("Verifying connectivity via target server ping handshake...")
            client.ping()
            
            # Commit state handles only after a successful handshake verification pass
            _redis_pool = pool
            _redis_client = client
            logger.info("Redis Core Client initialized and verified successfully.")
            
        except (redis.RedisError, redis.ConnectionError) as conn_err:
            logger.error(f"Verification ping failed during connect(). Leaving client uninitialized: {str(conn_err)}")
            _redis_pool = None
            _redis_client = None
        except Exception as e:
            logger.error(f"Unexpected exception building Redis connection matrix payload structures: {str(e)}")
            _redis_pool = None
            _redis_client = None


def disconnect() -> None:
    """Gracefully closes the active Redis client connections and pool resources 

    to ensure clean context disposal during application container shutdowns.
    """
    global _redis_client, _redis_pool

    with _init_lock:
        try:
            if _redis_client:
                _redis_client.close()
                logger.info("Redis network client execution layer closed.")

            if _redis_pool:
                _redis_pool.disconnect()
                logger.info("Redis connection pool resources detached completely.")

            logger.info("Redis connection pool shutdown complete.")
        except Exception as e:
            logger.warning(f"Redis shutdown boundary encountered cleanup warnings: {str(e)}")
        finally:
            _redis_client = None
            _redis_pool = None


def reconnect() -> None:
    """Forces an explicit tear-down and rebuild of the singleton connection state 
    to assist with automatic self-healing routines if network gaps occur.
    """
    logger.info("Triggering explicit network pipeline recovery sequence...")
    disconnect()
    connect()


def _get_client() -> Optional[redis.Redis]:
    """Internal runtime accessor handle. Lifecycle is controlled via 
    FastAPI startup/lifespan events.
    """
    return _redis_client


def generate_cache_key(question: str) -> Optional[str]:
    """Formulates deterministic namespace-grouped SHA-256 string signatures.
    
    Normalizes multi-spaces and converts case structures to ensure consistency.
    """
    if not question or not question.strip():
        return None
        
    try:
        normalized = " ".join(question.lower().split())
        hash_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"appna_bank:cache:{hash_digest}"
    except Exception as e:
        logger.error(f"Error executing key fingerprint compression: {str(e)}")
        return None


def get(key: str) -> Optional[Dict[str, Any]]:
    """Reads and transparently parses structured payload items directly from memory.
    
    Gracefully handles errors to ensure Redis issues do not break the main app pipeline.
    """
    if not key:
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        cached_data = client.get(key)
        if cached_data:
            logger.info("Cache hit occurred on current signature layout footprint tracking.")
            return json.loads(cached_data)
            
        logger.info("Cache miss recorded against target tracking matrix block.")
        return None
    except Exception as e:
        logger.error(f"Graceful Error Handling: Intercepted GET query mapping exception: {str(e)}")
        return None


def set(key: str, value: Dict[str, Any], ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    """Serializes standardized responses safely down into cache slots using setex().
    
    Validates dictionary elements strictly against specification rules before updating keys.
    """
    if not key or not value:
        return False

    client = _get_client()
    if client is None:
        return False

    # Base dictionary integrity verification checks
    if not isinstance(value, dict) or "answer" not in value or not str(value.get("answer")).strip():
        logger.warning("Cache set rejected: Supplied data payload structural mismatch errors caught.")
        return False

    try:
        # Standardize the cached payload format explicitly
        payload = {
            "answer": value.get("answer"),
            "detected_language": value.get("detected_language", "English"),
            "model": value.get("model", "qwen/qwen3-32b"),
            "source": value.get("source", "Groq"),
            "success": value.get("success", True),
            "cached_at": datetime.now(timezone.utc).isoformat()
        }
        
        serialized_string = json.dumps(payload)
        
        # Use setex explicitly to enforce set-with-expiration behavior
        client.setex(key, ttl, serialized_string)
        logger.info(f"Cache entry successfully committed. Allocated expiration window TTL: {ttl}s")
        return True
    except Exception as e:
        logger.error(f"Graceful Error Handling: Intercepted SET query transaction exception: {str(e)}")
        return False


def exists(key: str) -> bool:
    """Asserts existence signatures directly within the cluster map to prevent redundant lookups."""
    if not key:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        return bool(client.exists(key))
    except Exception as e:
        logger.error(f"Failed querying cache address key signature assertion state: {str(e)}")
        return False


def ttl(key: str) -> int:
    """Retrieves remaining time-to-live seconds for a specific key.
    Returns -1 if the key exists but has no associated expire, and -2 if the key does not exist.
    """
    if not key:
        return -2

    client = _get_client()
    if client is None:
        return -2

    try:
        return int(client.ttl(key))
    except Exception as e:
        logger.error(f"Failed pulling remaining lifespan analytics for tracking signature block: {str(e)}")
        return -2


def delete(key: str) -> bool:
    """Evicts a target signature block payload directly out of the memory cluster map."""
    if not key:
        return False

    client = _get_client()
    if client is None:
        return False

    try:
        result = client.delete(key)
        logger.info(f"Cache key entry invalidation run executed. Clean parameters cleared count: {result}")
        return result > 0
    except Exception as e:
        logger.error(f"Error executing explicit signature key erasure commands: {str(e)}")
        return False


def clear() -> bool:
    """Safely sweeps and clears the explicit appna_bank namespace cluster block via scan sweeps."""
    client = _get_client()
    if client is None:
        return False

    try:
        cursor = 0
        match_pattern = "appna_bank:cache:*"
        total_evicted = 0
        
        while True:
            cursor, keys = client.scan(cursor=cursor, match=match_pattern, count=100)
            if keys:
                client.delete(*keys)
                total_evicted += len(keys)
            if cursor == 0:
                break
                
        logger.info(f"Cache cleanup operations completed execution. Evicted namespace index maps count: {total_evicted}")
        return True
    except Exception as e:
        logger.error(f"Fatal exception occurred executing namespace cache purge routines: {str(e)}")
        return False


def health_check() -> Dict[str, Any]:
    """Returns extended connectivity status validation mapping diagnostics for FastAPI health monitoring."""
    start_time = time.time()
    is_reachable = False
    diagnostic_message = "Redis cache infrastructure operational."
    key_count = 0

    try:
        client = _get_client()
        if client is not None:
            client.ping()
            is_reachable = True
            _, keys = client.scan(cursor=0, match="appna_bank:cache:*", count=1000)
            key_count = len(keys)
        else:
            diagnostic_message = "Client execution context uninitialized or currently offline."
    except Exception as e:
        diagnostic_message = f"Health check connection handshake failure: {str(e)}"

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "healthy" if is_reachable else "unhealthy",
        "redis_connected": is_reachable,
        "redis_url_target": REDIS_URL.split("@")[-1],
        "cached_keys_estimated_count": key_count,
        "latency_ms": elapsed_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": diagnostic_message
    }
