from typing import TypedDict


class AMIAState(TypedDict):
    # Input
    query: str
    conversation_history: list[dict]   # [{"role": "user", "content": "..."}]

    # Llenado por DocExpert (V1)
    retrieved_docs: list[dict]         # chunks con text, metadata, scores

    # Llenado por SensorAnalyst (V2) y EconomicAnalyst (V3)
    sensor_analysis: dict | None
    economic_impact: dict | None
    work_order: dict | None

    # Decisión del Supervisor → qué nodo ejecutar a continuación
    next_agent: str

    # Output final
    final_response: str
    sources: list[dict]
