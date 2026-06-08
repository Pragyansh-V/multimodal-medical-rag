# Multimodal Medical RAG

A production-grade Retrieval-Augmented Generation pipeline for medical Q&A — combining pathology image retrieval with vision-language model answering.

> **"Ask a clinical question. Get an answer grounded in real pathology images."**

---

## What This Does

```
User question
      ↓
Embed with MiniLM (text) or CLIP text encoder (image/hybrid)
      ↓
ChromaDB retrieves top-k similar pathology images + Q&A pairs
      ↓
Retrieved images + context sent to Gemini 2.5 Flash
      ↓
VLM generates a grounded medical answer
```

---

## Architecture

```
KAGGLE GPU (offline — one-time setup)        LOCAL (serving)
──────────────────────────────────           ──────────────────────────
PathVQA dataset (HuggingFace)                FastAPI server
      ↓                                            ↓
CLIP ViT-L/14 → image embeddings (768-dim)   app/retriever.py
      ↓                                       ├── MiniLM (text queries)
MiniLM → text embeddings (384-dim)           └── CLIP text encoder
      ↓                                            (image/hybrid queries)
ChromaDB PersistentClient                          ↓
      ↓                                       app/vlm.py
Export index + images + metadata              └── Gemini 2.5 Flash
```

---

## Dataset

**PathVQA** — `flaviagiammarino/path-vqa` (HuggingFace)

| Split | Examples |
|---|---|
| Train | 19,654 |
| Validation | 6,259 |
| Test | 6,719 |

This project uses a **balanced 500-example sample** (250 yes/no + 250 open-ended) from the training split for fast iteration.

---

## Retrieval Modes

| Mode | Query embedding | Collection | Use case |
|---|---|---|---|
| `text` | MiniLM 384-dim | text | Semantic question similarity |
| `image` | CLIP text encoder 768-dim | image | Cross-modal text→image retrieval |
| `hybrid` | Both | both (merged) | Best of both modalities |

---

## API

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Root |
| `GET` | `/health` | Component status |
| `POST` | `/query` | Multimodal RAG query |

### Sample Request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Is there evidence of inflammation in this tissue?",
    "top_k": 3,
    "retrieval_mode": "hybrid"
  }'
```

### Sample Response

```json
{
  "question": "Is there evidence of inflammation in this tissue?",
  "answer": "Yes, there is clear evidence of inflammation — numerous inflammatory cells with small dark blue nuclei infiltrating between muscle fibers.",
  "retrieved_contexts": [
    {
      "id": "pathvqa_166",
      "question": "what contains epithelioid cell granulomas with caseation necrosis?",
      "answer": "interstitium",
      "similarity_score": 0.5488,
      "image_path": "data/sample_images/pathvqa_166.jpg"
    }
  ],
  "retrieval_mode": "hybrid",
  "model_used": "gemini-2.5-flash",
  "num_contexts": 3
}
```

---

## Project Structure

```
multimodal-medical-rag/
├── app/
│   ├── main.py          # FastAPI server — lifespan, routes
│   ├── retriever.py     # ChromaDB + CLIP + MiniLM retrieval
│   ├── vlm.py           # Gemini 2.5 Flash VLM answering
│   └── schemas.py       # Pydantic request/response schemas
├── kaggle/
│   └── embed_pathvqa.ipynb  # GPU notebook — embeddings + index
├── scripts/
│   └── test_gemini.py   # Gemini API sanity check
├── data/
│   ├── chroma_index/    # ChromaDB vector index (gitignored)
│   ├── sample_images/   # 500 PathVQA images (gitignored)
│   └── metadata.json    # Q&A metadata (gitignored)
├── tests/
├── .env.example
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Kaggle account with GPU access (for index generation)

### Local Setup

```bash
git clone https://github.com/Pragyansh-V/multimodal-medical-rag.git
cd multimodal-medical-rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Generate the Index (Kaggle)

1. Open `kaggle/embed_pathvqa.ipynb` in Kaggle
2. Enable GPU T4 x2 accelerator
3. Run all cells — generates embeddings for 500 PathVQA examples
4. Download `multimodal_rag_index.zip`, `sample_images.zip`, `metadata.json`
5. Extract into `data/` directory

### Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the Swagger UI.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| CLIP ViT-L/14 | Image embeddings (768-dim) |
| all-MiniLM-L6-v2 | Text embeddings (384-dim) |
| ChromaDB | Vector store — persistent, local |
| Gemini 2.5 Flash | Vision-language answering |
| FastAPI | API server |
| PathVQA | Pathology image-question dataset |
| Kaggle T4 GPU | Embedding generation |

---

## Key Design Decisions

**Why two embedding models?** CLIP and MiniLM serve different purposes — CLIP enables cross-modal text→image retrieval (768-dim shared space), MiniLM gives stronger semantic text similarity (384-dim, faster). Hybrid mode combines both for best coverage.

**Why Gemini 2.5 Flash?** Best free-tier vision model available — 1500 requests/day, strong medical image understanding, 1M token context window. Swappable via one config change.

**Why 500 examples?** Sufficient to demonstrate the full pipeline with meaningful retrieval diversity. The Kaggle notebook is parameterised — scale to 5000+ by changing one variable.

---

## Portfolio Context

Builds on:
- [Project 3 — RAG Pipeline](https://github.com/Pragyansh-V) — extended to multimodal
- [GateKeeper Medical AI](https://github.com/Pragyansh-V) — medical domain continuity
- [Project 7 — MLflow Tracking](https://github.com/Pragyansh-V/mlflow-experiment-tracking) — same Wisconsin Breast Cancer domain