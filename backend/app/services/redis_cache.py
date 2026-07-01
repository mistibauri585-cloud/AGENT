import os
import redis
import logging

logger = logging.getLogger(__name__)

redis_client = None


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
