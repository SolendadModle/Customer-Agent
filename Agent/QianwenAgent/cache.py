"""
缓存模块
提供简单的内存 LRU 缓存，减少重复 API 调用开销
"""

import time
from collections import OrderedDict
from typing import Any, Optional


class MemoryCache:
    """线程安全的内存 LRU 缓存"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Args:
            max_size: 最大缓存条数
            ttl: 缓存存活时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，不存在或已过期返回 None"""
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        # LRU: 移到末尾
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        """设置缓存值"""
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (value, time.time())
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def delete(self, key: str):
        """删除缓存条目"""
        self._cache.pop(key, None)

    def clear(self):
        """清空缓存"""
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


# 全局缓存实例
_cache = MemoryCache()


def get_cache() -> MemoryCache:
    """获取全局缓存实例"""
    return _cache
