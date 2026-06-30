# backend/app/main.py
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core Architectural Routing and Database Imports
from app.api.endpoints import router
from app.services.rag.pipeline import ingest_all_pdfs_from_folder
from app.services import redis_cache
from app.database import chromadb_client

# 7. Configure isolated module-scoped logger (Without basicConfig overrides)
logger = logging.getLogger(__name__)

# =====================================================================
# 4. LIFESPAN WORKSPACE ENGINE (PROPER MODERN FASTAPI INITIALIZATION)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 8. Start exact visual log sequence and initialize startup benchmark timer
    logger.info("---------------------------------------------------")
    logger.info("Starting Appna Finance AI Backend...")
    startup_start = time.time()
    
    # --- STEP 1: INITIALIZE REDIS ---
    logger.info("Initializing Redis...")
    try:
        # 4 & 9. Safe execution connection attempt
        redis_cache.connect()
        logger.info("Redis Connected")
    except Exception as e:
        logger.error(f"Redis initialization failed: {str(e)}", exc_info=True)
        logger.warning("Continuing deployment without active Redis cache tier.")

    # --- STEP 2: INITIALIZE CHROMADB ---
    logger.info("Initializing ChromaDB...")
    try:
        # 4 & 10. Isolated database initialization execution
        chromadb_client.initialize_database()
        logger.info("ChromaDB Ready")
    except Exception as e:
        logger.error(f"ChromaDB database layer setup failed: {str(e)}", exc_info=True)

    # --- STEP 3: LOAD KNOWLEDGE BASE COLLECTION ---
    logger.info("Checking Knowledge Base...")
    try:
        collection = chromadb_client.get_collection("appna_bank_knowledge")
        
        # 4. Evaluate document payload caching matrix bounds
        if collection is not None and collection.count() == 0:
            logger.info("Knowledge Base Empty → Auto Index Started")
            try:
                # 11. Run safe local indexing automation script 
                status = ingest_all_pdfs_from_folder("pdfs")
                logger.info(f"Auto-index sequence finished: {status}")
            except Exception as index_err:
                logger.error(f"PDF indexing automation pipeline failed: {str(index_err)}", exc_info=True)
        else:
            count = collection.count() if collection is not None else 0
            logger.info(f"Existing knowledge base detected. Loaded {count} cached data points.")
            
    except Exception as e:
        logger.error(f"Error validating knowledge base metrics status: {str(e)}", exc_info=True)

    startup_time = round(time.time() - startup_start, 2)
    logger.info(f"Startup Complete - Backend started successfully in {startup_time} seconds.")
    logger.info("---------------------------------------------------")
    
    yield  # Hand over operational runtime flow control to the ASGI application layer
    
    # =====================================================================
    # GRACEFUL SHUTDOWN SEQUENCING
    # =====================================================================
    logger.info("Stopping Appna Finance AI Backend...")
    
    try:
        if hasattr(redis_cache, 'disconnect'):
            redis_cache.disconnect()
            logger.info("Redis connection pool disconnected cleanly.")
        elif hasattr(redis_cache, 'redis_client') and hasattr(redis_cache.redis_client, 'close'):
            redis_cache.redis_client.close()
            logger.info("Redis network client closed cleanly.")
    except Exception as shutdown_err:
        logger.warning(f"Redis cleanup boundary encountered warning: {str(shutdown_err)}")
        
    logger.info("Shutdown complete. Container terminating safely.")


# 1. Initialize core FastAPI framework using the standard lifespan routing agent
app = FastAPI(
    title="Appna Bank AI - Backend MVP", 
    version="1.0",
    lifespan=lifespan
)

# 2. Keep the existing communication access layers (CORS Middleware Configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Maintain existing functional API routes map binding instance
app.include_router(router)

# =====================================================================
# 12. SYSTEM APPLICATION STATUS & DIAGNOSTICS ENDPOINTS
# =====================================================================
@app.get("/")
def root():
    # 12. Return the exact standardized structural analytics payload frame
    return {
        "message": "Appna Finance AI Backend is running.",
        "version": "1.0",
        "status": "healthy"
    }


@app.get("/health")
def health():
    """Dedicated health check endpoint providing structural diagnostic payloads 
    for monitoring dashboards, verifying connectivity status across database layers.
    """
    redis_healthy = False
    redis_diagnostics = "disconnected"
    
    # 1. Safely evaluate Redis connectivity status
    try:
        if hasattr(redis_cache, 'check_connection'):
            redis_diagnostics = redis_cache.check_connection()
            # Handle both boolean or dictionary response representations smoothly
            if isinstance(redis_diagnostics, dict):
                redis_healthy = redis_diagnostics.get("status") == "healthy"
            else:
                redis_healthy = bool(redis_diagnostics)
        elif hasattr(redis_cache, 'redis_client'):
            redis_healthy = bool(redis_cache.redis_client.ping())
            redis_diagnostics = "connected" if redis_healthy else "disconnected"
    except Exception as redis_err:
        redis_diagnostics = {"status": "unhealthy", "error": str(redis_err)}
        redis_healthy = False

    # 2. Safely evaluate ChromaDB connectivity and extract nested schema state
    chroma_status = {"status": "unhealthy", "error": "Unknown status check configuration"}
    chroma_healthy = False
    
    try:
        if hasattr(chromadb_client, 'check_database'):
            chroma_status = chromadb_client.check_database()
            # FIXED: Explicitly string-match the status instead of evaluating the truthiness of the dict
            chroma_healthy = chroma_status.get("status") == "healthy"
        else:
            # Safe structural fallback sequence if interface contract shifts
            collection_exists = chromadb_client.get_collection("appna_bank_knowledge") is not None
            chroma_status = {"status": "healthy" if collection_exists else "unhealthy"}
            chroma_healthy = collection_exists
    except Exception as chroma_err:
        chroma_status = {"status": "error", "error": str(chroma_err)}
        chroma_healthy = False

    # 3. Compute overall system state based on strict component health criteria
    system_status = "healthy" if (redis_healthy and chroma_healthy) else "degraded"

    return {
        "status": system_status,
        "redis": redis_diagnostics,
        "chromadb": chroma_status,
        "version": "1.0",
        "timestamp": time.time()
    }
