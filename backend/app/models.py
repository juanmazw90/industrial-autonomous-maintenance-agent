

import uuid

from pydantic import BaseModel, Field


class Documents(BaseModel):
    id: int
    title: str
    filename: str
    page_count: int
    chunk_count: int
    created_at: str

class DocumentsChunks(BaseModel):
    id: int
    document_id: int
    chunk_index: int
    content: str
    embedding: list[float]
    page_number: int
    metadata: dict


class InputQuery(BaseModel):
    query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

