from app.services.api_key_manager import SupabaseKeyManager

# Singleton instance shared across the application
key_manager = SupabaseKeyManager()

__all__ = ["key_manager"]
