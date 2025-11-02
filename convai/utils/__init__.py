from datetime import datetime, timezone


def get_current_time() -> datetime:
    """
    Get current time
    """
    return datetime.now(timezone.utc)


__all__ = [
    "get_current_time",
]
