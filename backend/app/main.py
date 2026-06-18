import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .models import InputQuery

from .services.rag_config import RAGConfig
from .services.retrieval import Retriever
from .services.agent import Generator
from .services.ingestion import IngestionPipeline

config = RAGConfig()
IngestionPipeline(config)
retriever = Retriever(config)
generator = Generator(config)



app = FastAPI(
    title="AMIA Backend",
    description="Autonomous Maintenance Intelligence Agent API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/process_input")
async def process_input(user_input: InputQuery) -> dict[str, str]:
    retrieved_chunks = await retriever.retrieve(user_input.query)
    
    response = await generator.generate(
        query=user_input.query,
        chunks=retrieved_chunks,
        
    )
    return {
        "response": response.answer,
        
    }


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
