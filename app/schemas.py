# app/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Input schema for multimodal RAG query."""
    question: str = Field(..., description="Clinical question to answer")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to retrieve")
    retrieval_mode: str = Field(
        default="text",
        description="Retrieval mode: 'text', 'image', or 'hybrid'"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What type of cells are visible in this pathology sample?",
                    "top_k": 3,
                    "retrieval_mode": "text"
                }
            ]
        }
    }


class RetrievedContext(BaseModel):
    """A single retrieved result from ChromaDB."""
    id: str
    question: str
    answer: str
    answer_type: str
    similarity_score: float
    image_path: str


class QueryResponse(BaseModel):
    """Output schema for multimodal RAG query."""
    question: str
    answer: str
    retrieved_contexts: list[RetrievedContext]
    retrieval_mode: str
    model_used: str
    num_contexts: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    chromadb_loaded: bool
    image_collection_count: int
    text_collection_count: int
    gemini_available: bool