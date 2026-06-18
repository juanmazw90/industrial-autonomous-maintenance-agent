

from pydantic import BaseModel


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
    
