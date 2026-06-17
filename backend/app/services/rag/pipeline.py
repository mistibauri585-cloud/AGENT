from pathlib import Path
from pypdf import PdfReader
from app.database.chromadb_client import get_collection


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
):
    """
    Split large text into smaller overlapping chunks.
    """

    chunks = []
    words = text.split()

    if not words:
        return chunks

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks


def ingest_all_pdfs_from_folder(folder_path: str = "pdfs"):
    """
    Read all PDFs and store chunks in ChromaDB.
    """

    folder = Path(folder_path)

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return "Created pdfs folder. Upload PDFs and run again."

    collection = get_collection("appna_bank_knowledge")

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        return "No PDF files found."

    total_chunks = 0

    for pdf_file in pdf_files:

        try:
            reader = PdfReader(str(pdf_file))

            full_text = ""

            for page in reader.pages:

                try:
                    text = page.extract_text()

                    if text:
                        full_text += text + "\n"

                except Exception as page_error:
                    print(
                        f"Page extraction error in "
                        f"{pdf_file.name}: {page_error}"
                    )

            if not full_text.strip():
                continue

            chunks = split_text_into_chunks(full_text)

            for idx, chunk in enumerate(chunks):

                chunk_id = f"{pdf_file.stem}_{idx}"

                try:
                    existing = collection.get(ids=[chunk_id])

                    if existing and existing["ids"]:
                        continue

                except Exception:
                    pass

                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": pdf_file.name,
                            "chunk": idx
                        }
                    ]
                )

                total_chunks += 1

        except Exception as e:
            print(
                f"Error processing "
                f"{pdf_file.name}: {e}"
            )

    return (
        f"Processed {len(pdf_files)} PDFs "
        f"into {total_chunks} chunks."
    )


def search_bookshelf(
    query: str,
    n_results: int = 5
):
    """
    Search ChromaDB knowledge base.
    """

    collection = get_collection(
        "appna_bank_knowledge"
    )

    try:

        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        if documents:

            context = "\n\n".join(documents)

            source = (
                metadatas[0].get("source")
                if metadatas
                else "PDF Base"
            )

            return {
                "found": True,
                "context": context,
                "source": source
            }

    except Exception as e:

        print(f"Search Error: {e}")

    return {
        "found": False,
        "context": "",
        "source": "Web Search"
    }
