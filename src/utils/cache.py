"""
Step-Based Disk Cache Manager.
Handles serialization and resumption of intermediate state checkpoints 
to accelerate re-execution across long-running optimization steps.
"""

import os
import pickle
from typing import Any, Optional
from config import CacheConfig


class CacheManager:
    """
    Manages loading and saving intermediate framework execution states using pickle.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.cfg = config or CacheConfig()
        if self.cfg.enable_cache:
            os.makedirs(self.cfg.cache_dir, exist_ok=True)

    def _get_filepath(self, key: str) -> str:
        """Constructs sanitized file path for cache key."""
        filename = f"{key.replace('/', '_').replace(' ', '_')}.pkl"
        return os.path.join(self.cfg.cache_dir, filename)

    def exists(self, key: str) -> bool:
        """
        Checks whether a valid cache file exists for given key.

        :param key: Unique identifier for cached step.
        :return: True if cached file exists and caching is enabled.
        """
        if not self.cfg.enable_cache:
            return False
        return os.path.isfile(self._get_filepath(key))

    def load(self, key: str) -> Optional[Any]:
        """
        Loads cached object from disk.

        :param key: Unique identifier for cached step.
        :return: Cached Python object or None if missing.
        """
        if not self.exists(key):
            return None
        filepath = self._get_filepath(key)
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            print(f"[CacheManager] Loaded step checkpoint: '{key}'")
            return data
        except Exception as e:
            print(f"[CacheManager] Failed loading cache for '{key}': {e}")
            return None

    def save(self, key: str, data: Any) -> None:
        """
        Saves object state to disk cache.

        :param key: Unique identifier for cached step.
        :param data: Object data to pickle.
        """
        if not self.cfg.enable_cache:
            return
        filepath = self._get_filepath(key)
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[CacheManager] Saved step checkpoint: '{key}'")
        except Exception as e:
            print(f"[CacheManager] Failed saving cache for '{key}': {e}")