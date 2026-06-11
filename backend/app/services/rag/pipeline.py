import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from app.database.chromadb_client import get_collection

# This model translates text into numbers so it understands English, Hindi, and Bengali at the same time
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    """Chops long book paragraphs into manageable cards."""
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_all_pdfs_from_folder(folder_path: str = "pdfs"):
    """Reads every single PDF file inside your folder and puts it into the database."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return "Created empty pdfs/ folder. Please drop your files there!"

    collection = get_collection("appna_bank_knowledge")
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        return "No PDFs found to read inside the pdfs folder."

    total_chunks = 0
    for file_name in pdf_files:
        path = os.path.join(folder_path, file_name)
        reader = PdfReader(path)
        
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        chunks = split_text_into_chunks(full_text)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_{idx}"
            # Generate numerical vector coordinates for the text chunk
            vector = embedding_model.encode(chunk).tolist()
            
            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": file_name}]
            )
            total_chunks += 1

    return f"Processed {len(pdf_files)} PDFs into {total_chunks} database blocks!"

def search_bookshelf(query: str, n_results: int = 3):
    """Searches the database bookshelf to find text that answers the user's question."""
    collection = get_collection("appna_bank_knowledge")
    query_vector = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    
    if results and results['documents'] and len(results['documents'][0]) > 0:
        distances = results['distances'][0] if 'distances' in results else [0.0]
        # Only accept the text if it is actually relevant (distance check)
        if distances[0] < 1.5:  
            return {
                "found": True,
                "context": "\n".join(results['documents'][0]),
                "source": results['metadatas'][0][0]['source'] if results['metadatas'] else "PDF Base"
            }
            
    return {"found": False, "context": "", "source": "Web Search"}
