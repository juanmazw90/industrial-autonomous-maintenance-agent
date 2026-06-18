

# prompts.py

SYSTEM_PROMPT = """
Eres un experto en mantenimiento industrial.
SOLO usa el contexto proporcionado y cita usando [1], [2], etc. para referenciar las fuentes.
Responde en el idioma del usuario.
"""

# def build_user_prompt(query: str, retrieved_chunks: list[str]) -> str:
#     context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks))
#     return f"""
# Usa el siguiente contexto para responder:

# {context}

# Pregunta: {query}
# """

