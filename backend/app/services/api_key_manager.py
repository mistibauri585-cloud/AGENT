import os
import logging
from typing import Optional
from datetime import datetime, timezone
from supabase import create_client, Client
import groq

# Use structural logging (Avoid calling basicConfig here so it doesn't collide with main.py)
logger = logging.getLogger(__name__)

class SupabaseKeyManager:
    def __init__(self):
        """
        Manages an infinite, cyclic rotating pool of API keys from Supabase.
        Keys are continuously balanced by usage recency (oldest used key picked first).
        """
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        
        if not supabase_url or not supabase_key:
            logger.critical("Missing Supabase credentials in environment variables!")
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
            logger.error("Supabase client is not initialized.")
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
                logger.info(f"Cyclic Loop -> Loaded oldest active key ID: {self._active_key_id}")
                
                # Instantly stamp the current time. This pushes this key to the back of the line 
                # for the next transaction, ensuring all keys share the traffic equally.
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
                    logger.error(f"Non-fatal error pushing key to back of usage queue: {str(update_err)}")
                
                return self._active_key
            else:
                logger.critical("!!! ALL REGISTERED API KEYS ARE EXHAUSTED !!! Waiting for manual dashboard reset.")
                self._active_key = None
                self._active_key_id = None
                return None
                
        except Exception as e:
            logger.error(f"Error executing dynamic key query rotation loop: {str(e)}")
            return None

    def get_active_key(self) -> Optional[str]:
        """Returns the currently loaded key. If empty or cleared, refreshes from DB queue."""
        if not self._active_key:
            return self.refresh_keys_from_db()
        return self._active_key

    def get_groq_client(self) -> Optional[groq.Groq]:
        """
        FIXED: Dynamically wraps the loaded active rotated key string into an initialized 
        Groq client instance requested by the text/voice generation layer.
        """
        api_key = self.get_active_key()
        if not api_key:
            logger.error("Cannot initialize Groq client: No active API keys available.")
            return None
        
        try:
            # Instantiates Groq SDK instance using current pool token parameter
            return groq.Groq(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed initialization wrapper layout of Groq client SDK: {str(e)}")
            return None

    def handle_quota_exhausted(self) -> Optional[str]:
        """
        Flags the broken key as 'exhausted' with a timestamp, clears local cache,
        and instantly pops the next waiting active key from the queue.
        """
        if not self.supabase or not self._active_key_id:
            logger.error("Cannot rotate key: Database offline or no tracking state loaded.")
            return None

        try:
            bad_id = self._active_key_id
            logger.warning(f"Quota Exhausted for Key ID [{bad_id}]. Drop-flagging in Supabase...")

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
            logger.error(f"Failed to isolate exhausted key status in Supabase: {str(e)}")
            self._active_key = None
            return None
