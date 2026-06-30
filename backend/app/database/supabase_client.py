import os
from supabase import create_client, Client

_supabase: Client | None = None


def get_supabase_client() -> Client:
    global _supabase

    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not url:
            raise RuntimeError("SUPABASE_URL environment variable is missing.")

        if not key:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY environment variable is missing.")

        _supabase = create_client(url, key)

    return _supabase
