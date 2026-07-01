import os
import json
import hashlib
import re
import redis
import logging

logger = logging.getLogger(__name__)

redis_client = None

DEFAULT_TTL = 3600  # 1 hour


def connect():
    global redis_client

    if redis_client is None:
        redis_client = redis.from_url(
            os.getenv("REDIS_URL"),
            decode_responses=True,
        )

    redis_client.ping()
    return redis_client


def disconnect():
    global redis_client

    if redis_client:
        try:
            redis_client.close()
        except Exception:
            pass

    redis_client = None


def generate_cache_key(question: str) -> str:
    """
    Generate a deterministic Redis cache key.
    """
    if not question:
        return None

    normalized = question.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    return f"chat:{digest}"


def get(key: str):
    global redis_client

    if redis_client is None:
        connect()

    value = redis_client.get(key)

    if value is None:
        return None

    return json.loads(value)


def set(key: str, value: dict, ttl: int = DEFAULT_TTL):
    global redis_client

    if redis_client is None:
        connect()

    redis_client.setex(
        key,
        ttl,
        json.dumps(value)
    )

    return True


def delete(key: str):
    global redis_client

    if redis_client is None:
        connect()

    redis_client.delete(key)


def health_check():
    return check_connection()


def check_connection():
    global redis_client

    try:
        if redis_client is None:
            return {
                "status": "disconnected"
            }

        redis_client.ping()

        return {
            "status": "healthy"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
