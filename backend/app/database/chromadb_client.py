import chromadb
import os

# This sets up a folder to save our data chunks permanently
db_path = os.getenv("CHROMA_DB_PATH", "/tmp/chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)

def get_collection(name: str):
    # This acts like fetching a specific binder out of our drawer
    return chroma_client.get_or_create_collection(name=name)
