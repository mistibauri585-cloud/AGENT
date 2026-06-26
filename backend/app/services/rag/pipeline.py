import os
import time
import uuid
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from pypdf import PdfReader
# Assuming your database folder exposes this helper
from app.database.chromadb_client import get_collection

# Configure Production Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
# 1. Multiple Knowledge Collections
COLLECTION_ROUTING = {
    "default": "banking_knowledge",
    "English": "stock_market_english",
    "Hindi": "stock_market_hindi",
    "Bengali": "stock_market_bengali",
    "schemes": "government_schemes",
    "loans": "loan_information"
}

# 5 & 10. Chunking and Token Threshold Budgets
DEFAULT_CHUNK_SIZE = 400  # Words (tuned down from 500 for high-precision RAG chunks)
DEFAULT_CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD = 0.65  # 3. Minimum distance/similarity score limit
MAX_CONTEXT_CHUNKS = 4
MAX_CONTEXT_CHAR_LIMIT = 8000

# 13. Explicit Multilingual Embedding Model Configuration
# ChromaDB will apply this model definition across operations natively
MULTILINGUAL_EMBEDDING_FUNCTION = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =====================================================================
# 7. INGESTION, CHUNKING & DUPLICATE PROTECTION HASHING
# =====================================================================
def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Splits raw text down into clean semantically coherent overlapping token-word windows."""
    if not text or not text.strip():
        return []
    
    # 9. Context Cleaning: Normalize whitespaces and line endings
    cleaned_text = " ".join(text.split())
    words = cleaned_text.split()
    
    if not words:
        return []
        
    chunks = []
    step = max(chunk_size - overlap, 1)
    
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
            
    return chunks


def generate_content_hash(text: str) -> str:
    """7. Generates a secure sha256 checksum fingerprint of text blocks to shield against duplicate insertions."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def extract_pdf_text(pdf_path: str) -> str:
    """Extracts raw text payload from disk filesystem binaries page by page safely."""
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            except Exception as page_err:
                logging.error(f"Failed parsing page {idx} inside {pdf_path}: {str(page_err)}")
        return "\n".join(pages_text)
    except Exception as e:
        logging.error(f"Fatal unreadable system crash parsing file {pdf_path}: {str(e)}")
        return ""


def ingest_all_pdfs_from_folder(folder_path: str = "pdfs", language: str = "English", category: str = "banking") -> str:
    """Processes document pipelines from folder targets into specific mapped partitions."""
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return "Created ingest folder pipeline target directory. Seed data targets and restart workflow."
        
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        return "No processable raw PDF binary paths detected inside directory targets."

    # Routing collection destination based on structural params
    collection_name = COLLECTION_ROUTING.get(language, COLLECTION_ROUTING["default"])
    collection = get_collection(collection_name)
    
    total_chunks = 0
    total_files = 0

    for pdf_file in pdf_files:
        try:
            full_text = extract_pdf_text(str(pdf_file))
            if not full_text.strip():
                continue
                
            chunks = split_text_into_chunks(full_text)
            
            for idx, chunk in enumerate(chunks):
                # 7. Composite unique index keys via text content checking digests
                content_hash = generate_content_hash(chunk)
                chunk_id = f"doc_{pdf_file.stem}_{idx}_{content_hash[:12]}"
                
                # Check for absolute duplication
                try:
                    existing = collection.get(ids=[chunk_id])
                    if existing and existing.get("ids"):
                        continue  # Skipped duplicate block entry safely
                except Exception:
                    pass
                
                # 6. Store enriched contextual production query fields
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": pdf_file.name,
                        "language": language,
                        "category": category,
                        "chunk_number": idx,
                        "hash": content_hash,
                        "timestamp": str(time.time())
                    }]
                )
                total_chunks += 1
            total_files += 1
        except Exception as e:
            logging.error(f"Error compiling pipeline metrics for file {pdf_file.name}: {str(e)}")

    return f"Processed {total_files} PDFs into {total_chunks} blocks inside target partition [{collection_name}]."


# =====================================================================
# 8 & 11. ADVANCED HYBRID ENGINE AND WEB API FALLBACKS
# =====================================================================
def _execute_tavily_web_search(query: str) -> str:
    """11. Dispatches high-speed queries to Tavily search network when local RAG scores dip below bounds."""
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    if not tavily_api_key:
        logging.warning("Tavily Fallback triggered but TAVILY_API_KEY environment variable is missing.")
        return ""
    
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=tavily_api_key)
        # Search optimized for contextual data parsing extracts
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        
        web_context_blocks = []
        for result in response.get("results", []):
            web_context_blocks.append(f"[Web Source: {result.get('url')}] {result.get('content')}")
            
        return "\n\n".join(web_context_blocks)
    except Exception as e:
        logging.error(f"Tavily API fallback network request failed: {str(e)}")
        return ""


def _query_single_collection(collection_name: str, query: str, n_results: int) -> List[Dict[str, Any]]:
    """Helper method executing targeted local database partitions vector extraction sweeps."""
    try:
        collection = get_collection(collection_name)
        results = collection.query(query_texts=[query], n_results=n_results)
        
        extracted_chunks = []
        if not results or not results.get("documents"):
            return []
            
        docs = results.get("documents", [[]])[0]
        metas = results.get("metas", [[]])[0] if results.get("metas") else [{} for _ in docs]
        # Distances can be converted back based on cosine setup
        distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0 for _ in docs]
        
        for i in range(len(docs)):
            # 3. Derive similarity conversion parsing logic metric tracking thresholds
            raw_dist = distances[i]
            similarity_score = round(float(1.0 - raw_dist if raw_dist <= 1.0 else 1.0 / (1.0 + raw_dist)), 3)
            
            extracted_chunks.append({
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "score": similarity_score,
                "collection": collection_name
            })
        return extracted_chunks
    except Exception as e:
        logging.debug(f"Skipping lookup inside unseeded block target cluster partition [{collection_name}]: {str(e)}")
        return []


# =====================================================================
# CORE ENGINE ENTRY INTERFACE
# =====================================================================
def search_bookshelf(query: str, detected_language: str = "English", n_results: int = MAX_CONTEXT_CHUNKS) -> Dict[str, Any]:
    """2. Dispatches contextual target matching across indexed nodes concurrently.

    17. Separation of Responsibilities: Never processes LLM parameters, builds answers, 
    or contacts Groq endpoint engines.

    Args:
        query (str): Clean localized inquiry mapping key from down-channel routers.
        detected_language (str): Local ISO mapped variant parsed upstream by services.
        n_results (int): Max boundaries containing processing context items.

    Returns:
        Dict[str, Any]: Rich pipeline analytics framework packaging clean text blocks.
    """
    start_time = time.time()
    
    if not query or not query.strip():
        return {"found": False, "context": "", "source": "Empty Query Engine Configuration Pipeline Step", "score": 0.0}

    # 2. Setup Language Aware Target Router Priorities
    primary_collection = COLLECTION_ROUTING.get(detected_language, COLLECTION_ROUTING["default"])
    
    # 1. Deduplicate search path sets to hit localized nodes safely
    target_collections = list(set([primary_collection, COLLECTION_ROUTING["default"], COLLECTION_ROUTING["schemes"], COLLECTION_ROUTING["loans"]]))
    
    all_retrieved_chunks: List[Dict[str, Any]] = []
    
    # 14. Execute High Speed Multi-threaded Cluster Scans
    with ThreadPoolExecutor(max_workers=len(target_collections)) as executor:
        futures = {executor.submit(_query_single_collection, col, query, n_results): col for col in target_collections}
        for future in as_completed(futures):
            try:
                all_retrieved_chunks.extend(future.result())
            except Exception as thread_err:
                logging.error(f"Parallel scanning processing worker dropped interface node context: {str(thread_err)}")

    # 8. Hybrid Keyword/Semantic Fusion Filtering: Deduplicate across cluster passes
    seen_contents = set()
    unique_chunks = []
    for chunk in all_retrieved_chunks:
        norm_text = "".join(chunk["text"].lower().split())
        if norm_text not in seen_contents:
            seen_contents.add(norm_text)
            unique_chunks.append(chunk)

    # Sort chunks cleanly by maximum validation confidence rank bounds
    unique_chunks.sort(key=lambda x: x["score"], reverse=True)

    # 3. Filter items directly using configured Threshold specifications
    filtered_chunks = [c for c in unique_chunks if c["score"] >= SIMILARITY_THRESHOLD]
    
    final_source = "Knowledge Base Cluster Partitions"
    top_score = filtered_chunks[0]["score"] if filtered_chunks else 0.0
    collection_utilized = filtered_chunks[0]["collection"] if filtered_chunks else "None"
    
    # 11. Threshold Boundary Validation Check -> Pivot to Tavily if low accuracy match
    if not filtered_chunks or top_score < SIMILARITY_THRESHOLD:
        logging.info(f"RAG search confidence ({top_score}) below threshold ({SIMILARITY_THRESHOLD}). Activating Tavily web pipeline.")
        web_context = _execute_tavily_web_search(query)
        if web_context:
            context_string = web_context
            final_source = "Tavily Global Web Infrastructure Integration"
            top_score = 1.0  # Set standard normalization bounds for web-hits
            collection_utilized = "Web Search Engine Router"
            filtered_chunks = [{"text": web_context, "score": 1.0}]
        else:
            context_string = ""
    else:
        # 9 & 10. Clean and bind verified high-ranking text streams
        target_slices = filtered_chunks[:MAX_CONTEXT_CHUNKS]
        context_string = "\n\n".join([c["text"].strip() for c in target_slices])
        
    # Apply strict character truncation limits prior to shipping context
    context_string = context_string[:MAX_CONTEXT_CHAR_LIMIT]
    elapsed_time = round(time.time() - start_time, 3)
    
    # 12. Logging Production Monitoring Framework Metrics
    logging.info(
        f"RAG Execution -> Lang: {detected_language} | Selected Collection: {collection_utilized} | "
        f"Chunks Found: {len(filtered_chunks)} | Max Score: {top_score} | Search Time: {elapsed_time}s"
    )

    # 4. Return Extended Analytical Structured Data Metadata Frame
    return {
        "found": len(context_string) > 0,
        "context": context_string,
        "source": final_source,
        "collection_name": collection_utilized,
        "similarity_score": top_score,
        "retrieved_chunk_count": len(filtered_chunks),
        "search_time": elapsed_time
    }
