
"""
capa de cache con redis para almacenar los resultados de
las consultas a la base de datos y a los modelos de lenguaje, 
con el fin de mejorar el rendimiento y reducir la latencia en 
las respuestas. La clase RAGCache se encargará de gestionar el 
almacenamiento y recuperación de datos en Redis, utilizando claves
generadas a partir de las consultas y los resultados obtenidos. 
Además, se implementarán mecanismos para manejar la expiración 
de los datos almacenados y asegurar la consistencia de la información.

"""

import hashlib
import json
import time

import redis.asyncio as redis

from .rag_config import RAGConfig, RAGResponse

class RAGCache:
    def __init__(self, config, RAGConfig):
        self.config = config
        self.redis = redis.Redis(
            host="localhost",
            port=config.redis_port,
            decode_responses = True)
            
        self._rag_config = RAGConfig
        self._misses = 0
    
    def _cache_key(self, query: str) -> str:
        normalized = query.strip().lower()
        return f"rag:response:{hashlib.md5(normalized.encode()).hexdigest()}"

    async def get(self, query: str) -> RAGResponse | None:
        key = self._cache_key(query)
        cached = await self.redis.get(key)
        if cached:
            self._hits += 1
            data = json.loads(cached)
            return RAGResponse(
                answer=data["answer"],
                sources=data["sources"],
                cached=True,
                latency_ms=0,
            )
        self._misses += 1
        return None

    async def set(self, query: str, response: RAGResponse):
        if not self.config.cache_enabled:
            return
        key = self._cache_key(query)
        data = json.dumps({
            "answer": response.answer,
            "sources": response.sources,
            "cached_at": time.time(),
        })
        await self.redis.setex(key, self.config.cache_ttl, data)

    async def invalidate_all(self):
        """Invalidate all cached responses (e.g., after re-ingestion)."""
        keys = []
        async for key in self.redis.scan_iter("rag:response:*"):
            keys.append(key)
        if keys:
            await self.redis.delete(*keys)
        return len(keys)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    async def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "cached_keys": await self.redis.dbsize(),
        }