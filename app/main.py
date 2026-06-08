# app/main.py

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import QueryRequest, QueryResponse, RetrievedContext, HealthResponse
from app.retriever import MultimodalRetriever
from app.vlm import VLMClient

# ── Global instances ──────────────────────────────────────────────────────────
retriever = None
vlm       = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load retriever and VLM on startup."""
    global retriever, vlm

    print("Starting up — loading retriever...")
    retriever = MultimodalRetriever()

    print("Starting up — loading VLM...")
    vlm = VLMClient()

    print("✅ All components loaded. Server ready.")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal Medical RAG API",
    description="""
    RAG pipeline for medical Q&A using PathVQA pathology images.

    - Dataset    : PathVQA (500 sampled examples)
    - Embeddings : CLIP ViT-L/14 (images) + all-MiniLM-L6-v2 (text)
    - Vector DB  : ChromaDB
    - VLM        : Gemini 2.5 Flash
    - Retrieval  : text | image | hybrid
    """,
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Multimodal Medical RAG API",
        "docs":    "/docs",
        "health":  "/health",
        "query":   "/query"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Returns server and component status."""
    return HealthResponse(
        status="healthy",
        chromadb_loaded=retriever is not None and retriever.is_loaded,
        image_collection_count=retriever.img_collection.count() if retriever else 0,
        text_collection_count=retriever.txt_collection.count() if retriever else 0,
        gemini_available=vlm is not None and vlm.is_available,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query(request: QueryRequest):
    """
    Answer a medical question using multimodal RAG.

    Steps:
    1. Embed question with MiniLM
    2. Retrieve top-k similar entries from ChromaDB
    3. Load retrieved pathology images
    4. Send images + context to Gemini 2.5 Flash
    5. Return answer + retrieved contexts
    """
    if retriever is None or not retriever.is_loaded:
        raise HTTPException(status_code=503, detail="Retriever not loaded")
    if vlm is None or not vlm.is_available:
        raise HTTPException(status_code=503, detail="VLM not available")

    try:
        # Step 1 + 2 — retrieve
        contexts = retriever.retrieve(
            question=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode
        )

        # Step 3 + 4 — generate answer
        answer = vlm.answer(
            question=request.question,
            retrieved_contexts=contexts
        )

        # Step 5 — format response
        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_contexts=[RetrievedContext(**ctx) for ctx in contexts],
            retrieval_mode=request.retrieval_mode,
            model_used=vlm.model,
            num_contexts=len(contexts)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))