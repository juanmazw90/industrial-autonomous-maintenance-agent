

import os
from dotenv import load_dotenv

#from agents import Agent, Runner, function_tool




from .rag_config import RAGConfig , RAGResponse
#from .tools import search_documentation
from .prompts import SYSTEM_PROMPT
from .retrieval import Retriever, RetrievedChunk
from openai import AsyncOpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

#tools = [search_documentation]

#agent = Agent(name = "MaintenanceAgent",
#            instructions = SYSTEM_PROMPT,
#            tools=tools
#            )


class Generator:
    def __init__(self, config:RAGConfig  ):
        self.config = config
        self.client = AsyncOpenAI(api_key = api_key)
        
    def _build_context(self, chunks:list[RetrievedChunk]):
        
        parts= []
        
        for i, chunk in enumerate(chunks , 1):
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"Source {i} ({source}):\n{chunk.text}\n")
        return "\n".join(parts)
    
    def _build_sources(self, chunks:list[RetrievedChunk]) -> list[dict]:
        return [
            {
               "index": i + 1,
               "source": c.metadata.get("source", "unknown"),
               "doc_id": c.metadata.get("doc_id", ""),
               "chunk_index": c.metadata.get("chunk_index", 0),
               "reranck_score": round(c.rerank_score, 4),
            }
            for i , c in enumerate(chunks)

        ]
       
    async def generate(
            self, 
            query:str,
            chunks:list[RetrievedChunk]) -> RAGResponse:

            context = self._build_context(chunks),
            response = await self.client.chat.completions.create(
                 model = self.config.llm_model,
                 messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"
                     },
                 ],

                 temperature = self.config.llm_temperature,
                 max_tokens = self.config.max_tokens,
            )
            
            return RAGResponse(
                answer=response.choices[0].message.content or "",
                sources=self._build_sources(chunks)
            )







        
        


        





