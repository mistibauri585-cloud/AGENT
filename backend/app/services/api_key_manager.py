# backend/app/services/api_key_manager.py
import logging
import threading
from datetime import datetime, timezone
from typing import Optional
import groq
# Assume your project's Supabase client generator import matches your internal pathing
from app.database.supabase_client import get_supabase_client 

logger = logging.getLogger(__name__)

class SupabaseKeyManager:
    """Manages Groq API keys dynamically from a Supabase database backend tracking table.
    Enables automatic rotation, state management, and thread-safe pool management.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._current_client: Optional[groq.Groq] = None
        self._active_key_id: Optional[str] = None
        self._current_raw_key: Optional[str] = None
        
        # Safely initialize Supabase client to prevent startup import crashes
        try:
            self.supabase = get_supabase_client()
        except Exception:
            logger.exception("Failed to initialize Supabase client. Manager entering degraded state.")
            self.supabase = None
        
        # Load initial key candidate context under thread-safe visibility bounds
        with self._lock:
            self._load_next_valid_key()

    @property
    def current_key_id(self) -> Optional[str]:
        """Exposes only the database row ID for logging, never exposing the plaintext API key."""
        with self._lock:
            return self._active_key_id

    def get_active_key_count(self) -> int:
        """Returns the absolute current count of active keys within the database."""
        if self.supabase is None:
            logger.warning("Supabase client is uninitialized. Cannot count active keys.")
            return 0
            
        try:
            response = self.supabase.table("groq_api_keys")\
                .select("id", count="exact")\
                .eq("status", "active")\
                .execute()
            return response.count if response.count is not None else 0
        except Exception:
            logger.exception("Failed counting active database pooled keys.")
            return 0

    def has_active_keys(self) -> bool:
        """Returns True if there is at least one active key left in the pool."""
        with self._lock:
            if self._active_key_id is not None:
                return True
        return self.get_active_key_count() > 0

    def _clear_current_state(self) -> None:
        """Private helper to clear local state variables. 
        Assumes the caller already securely holds self._lock.
        """
        self._current_client = None
        self._active_key_id = None
        self._current_raw_key = None

    def _load_next_valid_key(self) -> None:
        """Queries the database for the oldest or next available active key profile row.
        Internal helper: Assumes the caller already securely holds self._lock.
        """
        if self.supabase is None:
            logger.warning("Supabase client is uninitialized. Aborting key loading pipeline.")
            self._clear_current_state()
            return

        try:
            # Predictably pulls the least recently used active key via ascending updated_at sequence
            response = self.supabase.table("groq_api_keys")\
                .select("id, api_key")\
                .eq("status", "active")\
                .order("updated_at", desc=False)\
                .limit(1)\
                .execute()
            
            if response.data and len(response.data) > 0:
                key_record = response.data[0]
                self._active_key_id = str(key_record["id"])
                self._current_raw_key = str(key_record["api_key"])
                self._current_client = groq.Groq(api_key=self._current_raw_key)
                logger.info(f"Successfully rotated to and loaded Key ID: {self._active_key_id}")
            else:
                self._clear_current_state()
                logger.warning("No active keys available within the Supabase connection pool storage table.")
        except Exception:
            logger.exception("Critical error encountered while fetching the next valid pooled API key profile structure.")
            self._clear_current_state()

    def cycle_key(self) -> None:
        """Pushes the current key to the back of the queue by updating its timestamp, 
        clears local state cache, and loads the next active key for true round-robin fallback.
        """
        with self._lock:
            if self.supabase is None:
                logger.warning("Supabase client is uninitialized. Cannot cycle key in remote database.")
                self._clear_current_state()
                return

            if self._active_key_id:
                target_key_id = self._active_key_id
                now_iso = datetime.now(timezone.utc).isoformat()
                logger.info(f"Cycling active runtime cache away from Key ID: {target_key_id} due to transient failure. Pushing to back of queue.")
                try:
                    self.supabase.table("groq_api_keys")\
                        .update({"updated_at": now_iso})\
                        .eq("id", target_key_id)\
                        .execute()
                except Exception:
                    logger.exception(f"Failed updating timestamp during cycle_key execution for Key ID: {target_key_id}")
                    
            self._clear_current_state()
            self._load_next_valid_key()

    def mark_current_key_used(self) -> None:
        """Updates the current key's timestamp to push it to the back of the active queue,
        then immediately rotates the in-memory client state to point to the next fresh candidate.
        """
        with self._lock:
            if self.supabase is None:
                logger.warning("Supabase client is uninitialized. Skipping usage timestamp sync.")
                self._clear_current_state()
                return

            if self._active_key_id:
                target_key_id = self._active_key_id
                now_iso = datetime.now(timezone.utc).isoformat()
                logger.info(f"Successfully completed transaction using Key ID: {target_key_id}. Moving key to back of fair-rotation queue and advancing cache.")
                try:
                    self.supabase.table("groq_api_keys")\
                        .update({"updated_at": now_iso})\
                        .eq("id", target_key_id)\
                        .execute()
                except Exception:
                    logger.exception(f"Failed to execute fair-rotation tracking update on Key ID: {target_key_id}")

            self._clear_current_state()
            self._load_next_valid_key()

    def handle_quota_exhausted(self) -> None:
        """Marks the current key as permanently exhausted inside the database using proper UTC 
        ISO string timestamps and forces an immediate rotation swap.
        """
        with self._lock:
            if self.supabase is None:
                logger.warning("Supabase client is uninitialized. Cannot mark key status as exhausted.")
                self._clear_current_state()
                return

            if not self._active_key_id:
                self._load_next_valid_key()
                return

            target_exhausted_id = self._active_key_id
            now_iso = datetime.now(timezone.utc).isoformat()
            logger.warning(f"Permanently marking Key ID: {target_exhausted_id} as exhausted/broken inside the database.")
            
            try:
                self.supabase.table("groq_api_keys")\
                    .update({
                        "status": "exhausted", 
                        "updated_at": now_iso,
                        "exhausted_at": now_iso
                    })\
                    .eq("id", target_exhausted_id)\
                    .execute()
            except Exception:
                logger.exception(f"Failed updating remote state context mapping criteria for Key ID: {target_exhausted_id}")

            # Invalidate current runtime instance memory block and proceed to load next active key
            self._clear_current_state()
            self._load_next_valid_key()

    def get_groq_client(self) -> Optional[groq.Groq]:
        """Returns the current valid operational client workspace wrapper instance."""
        with self._lock:
            # Optimized: Streamlined to avoid nested lock queries and redundant database counts
            if self._current_client is None:
                self._load_next_valid_key()
            return self._current_client
