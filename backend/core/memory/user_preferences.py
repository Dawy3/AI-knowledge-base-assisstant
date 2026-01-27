"""
User Preferences Storage.

FOCUS: Store user preferences, domain context
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """
    User preferences and context.
    
    Usage:
        prefs = UserPreferences(user_id="123")
        prefs.set("language", "en")
        prefs.set("expertise_level", "advanced")
        
        lang = prefs.get("language")
    """
    
    user_id: str
    preferences: dict = field(default_factory=dict)
    
    # Common preference keys
    LANGUAGE = "language"
    EXPERTISE_LEVEL = "expertise_level"  # beginner, intermediate, advanced
    RESPONSE_STYLE = "response_style"  # concise, detailed
    DOMAIN = "domain"  # user's domain/industry
    TIMEZONE = "timezone"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get preference value."""
        return self.preferences.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set preference value."""
        self.preferences[key] = value
    
    def delete(self, key: str) -> None:
        """Delete preference."""
        self.preferences.pop(key, None)
    
    def get_all(self) -> dict:
        """Get all preferences."""
        return self.preferences.copy()
    
    def to_dict(self) -> dict:
        """Serialize preferences."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserPreferences":
        """Deserialize preferences."""
        return cls(
            user_id=data["user_id"],
            preferences=data.get("preferences", {}),
        )


class UserPreferencesStore:
    """
    Persistent storage for user preferences.
    
    Usage:
        store = UserPreferencesStore(redis_client)
        
        prefs = store.get("user123")
        prefs.set("language", "en")
        store.save(prefs)
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        prefix: str = "user_prefs",
        ttl_seconds: int = 86400 * 30,  # 30 days
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.ttl = ttl_seconds
        
        # In-memory fallback
        self._cache: dict[str, UserPreferences] = {}
    
    def get(self, user_id: str) -> UserPreferences:
        """Get user preferences, create if not exists."""
        # Try Redis
        if self.redis:
            data = self.redis.get(f"{self.prefix}:{user_id}")
            if data:
                return UserPreferences.from_dict(json.loads(data))
        
        # Try cache
        if user_id in self._cache:
            return self._cache[user_id]
        
        # Create new
        return UserPreferences(user_id=user_id)
    
    def save(self, prefs: UserPreferences) -> None:
        """Save user preferences."""
        data = json.dumps(prefs.to_dict())
        
        if self.redis:
            self.redis.setex(
                f"{self.prefix}:{prefs.user_id}",
                self.ttl,
                data,
            )
        
        self._cache[prefs.user_id] = prefs
    
    def delete(self, user_id: str) -> None:
        """Delete user preferences."""
        if self.redis:
            self.redis.delete(f"{self.prefix}:{user_id}")
        
        self._cache.pop(user_id, None)