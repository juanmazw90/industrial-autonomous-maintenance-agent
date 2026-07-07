"""
run_golden_set.py — Evaluación batch del agente AMIA contra el dataset dorado.

Para cada pregunta en eval/golden_set.json:
  1. Envía la query al endpoint /process_input/stream del backend.
  2. Acumula la respuesta completa del agente.
  3. (Opcional) Usa Claude como LLM-as-judge para evaluar groundedness,
     hallucination_flag y accuracy comparando contra los reference_points.
  4. Guarda resultados en data/evaluation/golden_set_results.json.

Uso:
  # Solo ejecutar y guardar respuestas (sin juicio LLM)
  uv run python eval/run_golden_set.py

  # Con LLM-as-judge (requiere ANTHROPIC_API_KEY en el entorno)
  uv run python eval/run_golden_set.py --judge

  # Solo preguntas de una categoría
  uv run python eval/run_golden_set.py --category rul --judge

Requisitos:
  - Backend corriendo en http://localhost:8000
  - pip install anthropic httpx  (ya en pyproject.toml del backend)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT      = Path(__file__).resolve().parent.parent
GOLDEN_SET     = REPO_ROOT / "eval" / "golden_set.json"
RESULTS_DIR    = REPO_ROOT / "data" / "evaluation"
BACKEND_URL    = os.getenv("AMIA_BACKEND_URL", "http://localhost:8000")
SESSION_ID     = str(uuid.uuid4())

JUDGE_MODEL    = os.getenv("AMIA_JUDGE_MODEL", "claude-sonnet-5")
JUDGE_SYSTEM   = """Eres un evaluador experto de agentes de mantenimiento industrial.
Evalúas la respuesta de un agente IA comparando con los reference_points esperados.
Responde SOLO con un JSON válido con exactamente estas claves:
{
  "groundedness": 0.0-1.0,       // qué tan bien fundamentada está la respuesta en hechos técnicos
  "hallucination_flag": true/false, // true si la respuesta inventa datos sin base
  "accuracy": 1-5,               // 1=incorrecta, 3=parcial, 5=precisa y completa
  "notes": "breve justificación"
}"""


# ── Backend streaming call ─────────────────────────────────────────────────────

async def call_agent(query: str, session_id: str, client: httpx.AsyncClient) -> dict:
    full_text  = ""
    agent_used = "synthesizer"
    sources    = []
    cached     = False

    async with client.stream(
        "POST",
        f"{BACKEND_URL}/process_input/stream",
        json={"query": query, "session_id": session_id},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        buffer = ""
        async for chunk in resp.aiter_text():
            buffer += chunk
            lines = buffer.split("\n")
            buffer = lines.pop()  # la última línea puede estar incompleta
            for line in lines:
                if not line.startswith("data: "):
                    continue
                try:
                    msg = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "token":
                    full_text += msg.get("content", "")
                elif msg.get("type") == "done":
                    agent_used = msg.get("agent_used", agent_used)
                    sources    = msg.get("sources", [])
                    cached     = msg.get("cached", False)
                elif msg.get("type") == "error":
                    raise RuntimeError(msg.get("message", "Error del agente"))

    return {
        "response":   full_text,
        "agent_used": agent_used,
        "sources":    sources,
        "cached":     cached,
    }


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

async def judge_response(
    query: str,
    response: str,
    reference_points: list[str],
    client: httpx.AsyncClient,
    api_key: str,
) -> dict:
    prompt = (
        f"PREGUNTA DEL USUARIO:\n{query}\n\n"
        f"RESPUESTA DEL AGENTE:\n{response}\n\n"
        f"REFERENCE POINTS (conceptos que debería cubrir una respuesta correcta):\n"
        + "\n".join(f"- {p}" for p in reference_points)
    )

    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": JUDGE_MODEL,
            "max_tokens": 512,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    raw = resp.json()["content"][0]["text"].strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


# ── Runner ────────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    with open(GOLDEN_SET, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        if not questions:
            print(f"⚠️  No hay preguntas con categoría '{args.category}'")
            return

    api_key = os.getenv("ANTHROPIC_API_KEY", "") if args.judge else ""
    if args.judge and not api_key:
        print("⚠️  ANTHROPIC_API_KEY no está configurada. Ejecutando sin juicio LLM.")
        args.judge = False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    passed = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        for q in questions:
            print(f"\n[{q['id']}] {q['query'][:80]}…")

            result: dict = {
                "id":             q["id"],
                "query":          q["query"],
                "category":       q["category"],
                "agent_expected": q["agent_expected"],
                "timestamp":      datetime.now(UTC).isoformat(),
                "session_id":     SESSION_ID,
            }

            try:
                answer = await call_agent(q["query"], SESSION_ID, client)
                result.update({
                    "response":   answer["response"],
                    "agent_used": answer["agent_used"],
                    "cached":     answer["cached"],
                    "sources":    answer["sources"],
                    "error":      None,
                })
                print(f"    → agente: {answer['agent_used']}  |  {len(answer['response'])} chars")

                if args.judge and answer["response"]:
                    try:
                        judgment = await judge_response(
                            q["query"], answer["response"],
                            q["reference_points"], client, api_key,
                        )
                        result["judgment"] = judgment
                        mark = "✅" if judgment["accuracy"] >= 4 and not judgment["hallucination_flag"] else "⚠️"
                        print(
                            f"    {mark} accuracy={judgment['accuracy']}/5  "
                            f"groundedness={judgment['groundedness']:.2f}  "
                            f"hallucination={judgment['hallucination_flag']}"
                        )

                        if judgment["accuracy"] >= 3 and not judgment["hallucination_flag"]:
                            passed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"    ⚠️  Error en LLM-judge: {e}")
                        result["judgment"] = {"error": str(e)}

            except Exception as e:
                result["error"] = str(e)
                print(f"    ❌ Error: {e}")
                failed += 1

            results.append(result)

    # Guardar resultados — pass_rate solo tiene sentido cuando hubo juez LLM
    pass_rate = round(passed / len(results), 3) if (args.judge and results) else None
    out = {
        "run_id":      SESSION_ID,
        "timestamp":   datetime.now(UTC).isoformat(),
        "backend_url": BACKEND_URL,
        "judge_used":  args.judge,
        "judge_model": JUDGE_MODEL if args.judge else None,
        "total":       len(results),
        "passed":      passed,
        "failed":      failed,
        "pass_rate":   pass_rate,
        "results":     results,
    }

    out_path = RESULTS_DIR / "golden_set_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"Total: {len(results)} | Pasadas: {passed} | Falladas: {failed}")
    if pass_rate is not None:
        print(f"Tasa de aprobación: {pass_rate*100:.1f}%")
    else:
        print("Tasa de aprobación: n/a (ejecuta con --judge para evaluar)")
    print(f"Resultados guardados en {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación batch del agente AMIA contra el dataset dorado")
    parser.add_argument("--judge", action="store_true",
                        help="Activa LLM-as-judge con Claude (requiere ANTHROPIC_API_KEY)")
    parser.add_argument("--category", type=str, default="",
                        help="Filtrar por categoría (ej. rul, diagnostico, modos_de_fallo)")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
