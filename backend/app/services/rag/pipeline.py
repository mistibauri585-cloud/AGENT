from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
from app.database.chromadb_client import get_collection


COLLECTION_NAME = "appna_bank_knowledge"


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.
    """

    if not text:
        return []

    words = text.split()

    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF.
    """

    try:
        reader = PdfReader(pdf_path)

        pages_text = []

        for page in reader.pages:
            try:
                text = page.extract_text()

                if text:
                    pages_text.append(text)

            except Exception as page_error:
                print(
                    f"Page extraction failed "
                    f"in {pdf_path}: {page_error}"
                )

        return "\n".join(pages_text)

    except Exception as e:
        print(f"PDF read error: {pdf_path} -> {e}")
        return ""


def ingest_all_pdfs_from_folder(
    folder_path: str = "pdfs"
) -> str:
    """
    Read all PDFs and store chunks in ChromaDB.
    """

    folder = Path(folder_path)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

        return (
            "Created pdfs folder. "
            "Upload PDFs and run again."
        )

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        return "No PDF files found."

    collection = get_collection(COLLECTION_NAME)

    total_chunks = 0
    total_files = 0

    for pdf_file in pdf_files:

        try:
            full_text = extract_pdf_text(
                str(pdf_file)
            )

            if not full_text.strip():
                print(
                    f"Skipping empty PDF: "
                    f"{pdf_file.name}"
                )
                continue

            chunks = split_text_into_chunks(
                full_text
            )

            for idx, chunk in enumerate(chunks):

                chunk_id = (
                    f"{pdf_file.stem}_{idx}"
                )

                try:
                    existing = collection.get(
                        ids=[chunk_id]
                    )

                    if existing.get("ids"):
                        continue

                except Exception:
                    pass

                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": pdf_file.name,
                            "chunk_number": idx
                        }
                    ]
                )

                total_chunks += 1

            total_files += 1

        except Exception as e:
            print(
                f"Failed processing "
                f"{pdf_file.name}: {e}"
            )

    return (
        f"Processed {total_files} PDFs "
        f"into {total_chunks} chunks."
    )


def search_bookshelf(
    query: str,
    n_results: int = 5
) -> Dict:
    """
    Search Appna Bank knowledge base.
    """

    if not query.strip():
        return {
            "found": False,
            "context": "",
            "source": "No Query"
        }

    collection = get_collection(
        COLLECTION_NAME
    )

    try:

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        documents = (
            results.get(
                "documents",
                [[]]
            )[0]
        )

        metadatas = (
            results.get(
                "metadatas",
                [[]]
            )[0]
        )

        if not documents:
            return {
                "found": False,
                "context": "",
                "source": "Knowledge Base"
            }

        context = "\n\n".join(documents)

        source = (
            metadatas[0].get("source")
            if metadatas
            else "Knowledge Base"
        )

        return {
            "found": True,
            "context": context,
            "source": source
        }

    except Exception as e:

        print(
            f"Knowledge Search Error: {e}"
        )

        return {
            "found": False,
            "context": "",
            "source": "Search Failed"
        }
