import os
import logging
from typing import Optional
from supabase import create_client, Client

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SupabaseKeyManager:
    def __init__(self):
        """
        Manages rotating API keys directly from a Supabase database.
        Allows the developer to manually replace or add keys via the Supabase dashboard
        with instant, zero-downtime updates on Railway.
        """
        # Fetch Supabase credentials securely from Railway Environment Variables
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        
        if not supabase_url or not supabase_key:
            logging.critical("Missing Supabase credentials in environment variables!")
            self.supabase: Optional[Client] = None
        else:
            self.supabase = create_client(supabase_url, supabase_key)
            
        self._active_key: Optional[str] = None
        self._active_key_id: Optional[int] = None

    def refresh_keys_from_db(self) -> Optional[str]:
        """
        Queries Supabase for the oldest key marked as 'active'.
        Runs on initial startup and whenever a key rotation is triggered.
        """
        if not self.supabase:
            logging.error("Supabase client is not initialized.")
            return None

        try:
            response = self.supabase.table("api_keys") \
                .select("id, key_value") \
                .eq("status", "active") \
                .order("id") \
                .limit(1) \
                .execute()

            data = response.data
            if data and len(data) > 0:
                self._active_key_id = data[0]["id"]
                self._active_key = data[0]["key_value"].strip()
                logging.info(f"Loaded working API Key from Supabase slot ID: {self._active_key_id}")
                return self._active_key
            else:
                logging.critical("!!! ALL API KEYS IN SUPABASE ARE EXHAUSTED !!!")
                self._active_key = None
                self._active_key_id = None
                return None
                
        except Exception as e:
            logging.error(f"Error reading keys from Supabase: {str(e)}")
            return None

    def get_active_key(self) -> Optional[str]:
        """Returns the current cached key. If empty, pulls freshest from DB."""
        if not self._active_key:
            return self.refresh_keys_from_db()
        return self._active_key

    def handle_quota_exhausted(self) -> Optional[str]:
        """
        Marks the active key as 'exhausted' in Supabase when a quota/rate error is hit,
        then instantly retrieves the next available active key.
        """
        if not self.supabase or not self._active_key_id:
            logging.error("Cannot rotate key: Database offline or no active key loaded.")
            return None

        try:
            bad_id = self._active_key_id
            logging.warning(f"Quota finished for Key ID [{bad_id}]. Updating Supabase status to 'exhausted'...")

            # Update the status field on Supabase
            self.supabase.table("api_keys") \
                .update({"status": "exhausted"}) \
                .eq("id", bad_id) \
                .execute()

            # Instantly pull the next available working key from the table
            return self.refresh_keys_from_db()

        except Exception as e:
            logging.error(f"Failed to update key status in Supabase: {str(e)}")
            self._active_key = None
            return None
