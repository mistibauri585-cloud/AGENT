import os
import tempfile
import logging
from typing import List

from supabase import create_client

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

BUCKET_NAME = "knowledge-base"

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY
)


def list_pdfs_in_bucket() -> List[str]:
    """
    Returns all PDF filenames stored in the knowledge-base bucket.
    """
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()

        pdfs = []

        for file in files:
            name = file.get("name", "")
            if name.lower().endswith(".pdf"):
                pdfs.append(name)

        logger.info(f"Found {len(pdfs)} PDFs in Supabase Storage.")

        return pdfs

    except Exception as e:
        logger.error(f"Unable to list PDFs: {e}")
        return []


def download_pdfs_from_supabase() -> List[str]:
    """
    Downloads all PDFs from Supabase Storage into /tmp/pdfs
    and returns their local paths.
    """

    local_folder = os.path.join(tempfile.gettempdir(), "pdfs")

    os.makedirs(local_folder, exist_ok=True)

    downloaded_files = []

    pdf_files = list_pdfs_in_bucket()

    for pdf_name in pdf_files:

        try:

            logger.info(f"Downloading {pdf_name}")

            data = supabase.storage.from_(BUCKET_NAME).download(pdf_name)

            local_path = os.path.join(local_folder, pdf_name)

            with open(local_path, "wb") as f:
                f.write(data)

            downloaded_files.append(local_path)

            logger.info(f"Downloaded -> {local_path}")

        except Exception as e:

            logger.error(f"Failed downloading {pdf_name}: {e}")

    return downloaded_files


def cleanup_temp_pdfs():
    """
    Deletes temporary downloaded PDFs.
    """

    folder = os.path.join(tempfile.gettempdir(), "pdfs")

    if not os.path.exists(folder):
        return

    for file in os.listdir(folder):

        try:

            os.remove(os.path.join(folder, file))

        except Exception:

            pass

    logger.info("Temporary PDFs removed.")
