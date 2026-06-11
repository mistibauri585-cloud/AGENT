from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.services.rag.pipeline import ingest_all_pdfs_from_folder
from app.database.chromadb_client import get_collection

app = FastAPI(title="Appna Bank AI - Backend MVP", version="1.0")

# Enable communication access layers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
def startup_event():
    print("Initializing system vectors and verification checks...")
    try:
        collection = get_collection("appna_bank_knowledge")
        if collection.count() == 0:
            print("ChromaDB library repository empty. Auto-indexing local documents...")
            status = ingest_all_pdfs_from_folder("pdfs")
            print(status)
        else:
            print(f"System ready with {collection.count()} cached data points.")
    except Exception as e:
        print(f"Startup database bypass error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Appna Bank AI Production System live. Head over to /docs to review schemas."}
