"""
conversation.py — Persistencia de historial de conversación en Redis.

Cada sesión se identifica con un session_id (UUID). El historial se almacena
como lista de mensajes en formato Anthropic: [{"role": "user"|"assistant", "content": str}].

Se guarda solo la query del usuario y la respuesta del asistente — no el contexto
RAG, para evitar que el historial crezca con texto de los documentos.
"""

import json

import redis

_MAX_TURNS = 10        # 10 turnos = 20 mensajes (user + assistant)
_TTL_SECONDS = 86_400  # 24 horas


class ConversationStore:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self._redis = redis.Redis(host=host, port=port, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"conv:{session_id}"

    def get_history(self, session_id: str) -> list[dict]:
        """Devuelve el historial de la sesión o lista vacía si no existe."""
        raw = self._redis.get(self._key(session_id))
        if raw is None:
            return []
        return json.loads(raw)

    def append_turn(self, session_id: str, user_query: str, assistant_response: str) -> None:
        """Añade un turno (user + assistant) y renueva el TTL."""
        history = self.get_history(session_id)
        history.append({"role": "user",      "content": user_query})
        history.append({"role": "assistant", "content": assistant_response})
        # Ventana deslizante: mantiene solo los últimos N turnos
        if len(history) > _MAX_TURNS * 2:
            history = history[-(  _MAX_TURNS * 2):]
        self._redis.setex(self._key(session_id), _TTL_SECONDS, json.dumps(history))

    def clear(self, session_id: str) -> None:
        """Elimina el historial de una sesión (útil para tests o reset explícito)."""
        self._redis.delete(self._key(session_id))
