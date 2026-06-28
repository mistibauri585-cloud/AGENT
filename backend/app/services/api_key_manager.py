import os
import logging
from typing import Optional
from datetime import datetime, timezone
from supabase import create_client, Client

# Configure production logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [Key Manager] - %(message)s")

class SupabaseKeyManager:
    def __init__(self):
        """
        Manages an infinite, cyclic rotating pool of API keys from Supabase.
        Keys are continuously balanced by usage recency (oldest used key picked first).
        """
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        
        if not supabase_url or not supabase_key:
            logging.critical("Missing Supabase credentials in environment variables!")
            self.supabase: Optional[Client] = None
        else:
            self.supabase = create_client(supabase_url, supabase_key)
            
        self._active_key: Optional[str] = None
        self._active_key_id: Optional[str] = None

    def refresh_keys_from_db(self) -> Optional[str]:
        """
        Pulls the oldest 'active' key from Supabase based on 'last_used_at'.
        This creates an infinite cyclic queue when exhausted keys are reset to active.
        """
        if not self.supabase:
            logging.error("Supabase client is not initialized.")
            return None

        try:
            # CRITICAL: Sorting by last_used_at ASC means the key that hasn't been used 
            # for the longest time rotates to the front of the line.
            response = self.supabase.table("groq_api_keys") \
                .select("id, api_key") \
                .eq("status", "active") \
                .order("last_used_at", ascending=True) \
                .limit(1) \
                .execute()

            data = response.data
            if data and len(data) > 0:
                self._active_key_id = data[0]["id"]
                self._active_key = data[0]["api_key"].strip()
                logging.info(f"Cyclic Loop -> Loaded oldest active key ID: {self._active_key_id}")
                
                # Instantly stamp the current time. This pushes this key to the back of the line 
                # for the next transaction, ensuring all 5 keys share the traffic equally.
                try:
                    self.supabase.table("groq_api_keys").update(
                        {
                            "last_used_at": datetime.now(timezone.utc).isoformat()
                        }
                    ).eq(
                        "id",
                        self._active_key_id
                    ).execute()
                except Exception as update_err:
                    logging.error(f"Non-fatal error pushing key to back of usage queue: {str(update_err)}")
                
                return self._active_key
            else:
                logging.critical("!!! ALL REGISTERED API KEYS ARE EXHAUSTED !!! Waiting for manual dashboard reset.")
                self._active_key = None
                self._active_key_id = None
                return None
                
        except Exception as e:
            logging.error(f"Error executing dynamic key query rotation loop: {str(e)}")
            return None

    def get_active_key(self) -> Optional[str]:
        """Returns the currently loaded key. If empty or cleared, refreshes from DB queue."""
        if not self._active_key:
            return self.refresh_keys_from_db()
        return self._active_key

    def handle_quota_exhausted(self) -> Optional[str]:
        """
        Flags the broken key as 'exhausted' with a timestamp, clears local cache,
        and instantly pops the next waiting active key from the queue.
        """
        if not self.supabase or not self._active_key_id:
            logging.error("Cannot rotate key: Database offline or no tracking state loaded.")
            return None

        try:
            bad_id = self._active_key_id
            logging.warning(f"Quota Exhausted for Key ID [{bad_id}]. Drop-flagging in Supabase...")

            # Take the key out of the active loop rotation instantly
            self.supabase.table("groq_api_keys").update(
                {
                    "status": "exhausted",
                    "exhausted_at": datetime.now(timezone.utc).isoformat()
                }
            ).eq(
                "id",
                bad_id
            ).execute()

            # Force clear local instance state so get_active_key() pulls a fresh row
            self._active_key = None
            self._active_key_id = None

            # Instantly step to the next available account line item
            return self.refresh_keys_from_db()

        except Exception as e:
            logging.error(f"Failed to isolate exhausted key status in Supabase: {str(e)}")
            self._active_key = None
            return None
