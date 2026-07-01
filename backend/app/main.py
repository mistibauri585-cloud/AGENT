# backend/app/main.py
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core Architectural Routing and Database Imports
from app.api.endpoints import router
from app.services import redis_cache
from app.database import chromadb_client

# Explicitly pull Supabase client capabilities securely
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Configure isolated module-scoped logger (Without basicConfig overrides)
logger = logging.getLogger(__name__)

# Global storage placeholder for the verified Supabase Client instance
supabase_client: Optional[Any] = None

# =====================================================================
# LIFESPAN WORKSPACE ENGINE (PROPER MODERN FASTAPI INITIALIZATION)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global supabase_client
    
    # Start exact visual log sequence and initialize startup benchmark timer
    logger.info("---------------------------------------------------")
    logger.info("Starting Appna Finance AI Backend...")
    startup_start = time.time()
    
    # --- STEP 1: INITIALIZE REDIS ---
    logger.info("Initializing Redis...")
    try:
        redis_cache.connect()
        logger.info("Redis Connected")
    except Exception as e:
        logger.error(f"Redis initialization failed: {str(e)}", exc_info=True)
        logger.warning("Continuing deployment without active Redis cache tier.")

    # --- STEP 2: INITIALIZE CHROMADB ---
    logger.info("Initializing ChromaDB...")
    try:
        chromadb_client.initialize_database()
        logger.info("ChromaDB Ready")
    except Exception as e:
        logger.error(f"ChromaDB database layer setup failed: {str(e)}", exc_info=True)

    # --- STEP 3: INITIALIZE SUPABASE (SINGLE SOURCE OF TRUTH) ---
    logger.info("Initializing Supabase Core Connection...")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key or create_client is None:
        logger.error("Supabase environment configuration tokens or client libraries are missing.")
        logger.warning("Continuing deployment without verified Supabase single source of truth connection.")
    else:
        try:
            # Connect and instantiate the client engine
            supabase_client = create_client(supabase_url, supabase_key)
            
            # Fixed: Using a real table query connection test matching service role requirements
            supabase_client.table("groq_api_keys").select("id").limit(1).execute()
            logger.info("Supabase Connected")
        except Exception as e:
            logger.error(f"Supabase connection validation failed: {str(e)}", exc_info=True)
            logger.warning("Continuing backend initialization execution with degraded configuration access pools.")

    # --- STEP 4: EVALUATE KNOWLEDGE BASE STATUS WITHOUT INGESTION BLOCKING ---
    logger.info("Checking Knowledge Base Partition Analytics...")
    try:
        collection = chromadb_client.get_collection("appna_bank_knowledge")
        
        if collection is not None and collection.count() == 0:
            logger.info("Knowledge Base Empty. Waiting for manual indexing.")
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


# Initialize core FastAPI framework using the standard lifespan routing agent
app = FastAPI(
    title="Appna Bank AI - Backend MVP", 
    version="1.0",
    lifespan=lifespan
)

# Keep the existing communication access layers (CORS Middleware Configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maintain existing functional API routes map binding instance
app.include_router(router)

# =====================================================================
# SYSTEM APPLICATION STATUS & DIAGNOSTICS ENDPOINTS
# =====================================================================
@app.get("/")
def root():
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
            chroma_healthy = chroma_status.get("status") == "healthy"
        else:
            collection_exists = chromadb_client.get_collection("appna_bank_knowledge") is not None
            chroma_status = {"status": "healthy" if collection_exists else "unhealthy"}
            chroma_healthy = collection_exists
    except Exception as chroma_err:
        chroma_status = {"status": "error", "error": str(chroma_err)}
        chroma_healthy = False

    # 3. Dynamic Supabase connectivity checking using the real data query format
    supabase_healthy = False
    try:
        if supabase_client is not None:
            supabase_client.table("groq_api_keys").select("id").limit(1).execute()
            supabase_healthy = True
    except Exception:
        pass

    # Compute overall system state based on strict component health criteria
    system_status = "healthy" if (redis_healthy and chroma_healthy) else "degraded"

    return {
        "status": system_status,
        "redis": redis_diagnostics,
        "chromadb": chroma_status,
        "supabase": "connected" if supabase_healthy else "disconnected",
        "version": "1.0",
        "timestamp": time.time()
    }
