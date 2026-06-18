# app/retriever.py

import os
import json
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable

import chromadb
import open_clip
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT  = Path(__file__).parent.parent
CHROMA_PATH   = PROJECT_ROOT / "data" / "chroma_index"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
IMAGES_PATH   = PROJECT_ROOT / "data" / "sample_images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MultimodalRetriever:

    def __init__(self):
        self.client         = None
        self.img_collection = None
        self.txt_collection = None
        self.text_model     = None
        self.clip_model     = None
        self.clip_tokenizer = None
        self.metadata       = None
        self._load()

    def _load(self):
        print("Loading ChromaDB...")
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.img_collection = self.client.get_collection("pathvqa_images")
        self.txt_collection = self.client.get_collection("pathvqa_text")
        print(f"  Image collection : {self.img_collection.count()} items")
        print(f"  Text collection  : {self.txt_collection.count()} items")

        print("Loading MiniLM for text retrieval...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  MiniLM loaded")

        print("Loading CLIP for image retrieval...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai', device=DEVICE
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        print("  CLIP ViT-L/14 loaded")

        print("Loading metadata...")
        with open(METADATA_PATH) as f:
            self.metadata = {item['id']: item for item in json.load(f)}
        print(f"  Metadata loaded: {len(self.metadata)} entries")

    def _embed_text_minilm(self, question: str) -> list[float]:
        """384-dim embedding — for text collection queries."""
        embedding = self.text_model.encode(
            [question], normalize_embeddings=True
        )
        return embedding[0].tolist()

    def _embed_text_clip(self, question: str) -> list[float]:
        """768-dim embedding — for image collection queries."""
        tokens = self.clip_tokenizer([question]).to(DEVICE)
        with torch.no_grad():
            embedding = self.clip_model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding[0].cpu().numpy().tolist()

    @traceable(name="retrieve", run_type="chain")
    def retrieve(self, question: str, top_k: int = 3, mode: str = "text") -> list[dict]:
        if mode == "text":
            return self._query_collection(
                self.txt_collection,
                self._embed_text_minilm(question),
                top_k
            )
        elif mode == "image":
            return self._query_collection(
                self.img_collection,
                self._embed_text_clip(question),
                top_k
            )
        elif mode == "hybrid":
            text_results  = self._query_collection(
                self.txt_collection,
                self._embed_text_minilm(question),
                top_k
            )
            image_results = self._query_collection(
                self.img_collection,
                self._embed_text_clip(question),
                top_k
            )
            return self._merge_results(text_results, image_results, top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @traceable(name="query_collection", run_type="retriever")
    def _query_collection(self, collection, query_embedding, top_k) -> list[dict]:
        response = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        results = []
        for i, (meta, dist) in enumerate(zip(
            response['metadatas'][0],
            response['distances'][0]
        )):
            doc_id     = response['ids'][0][i]
            similarity = round(1 - dist, 4)
            results.append({
                "id":               doc_id,
                "question":         meta['question'],
                "answer":           meta['answer'],
                "answer_type":      meta['answer_type'],
                "similarity_score": similarity,
                "image_path":       str(IMAGES_PATH / f"{doc_id}.jpg"),
            })
        return results

    @traceable(name="merge_results", run_type="chain")
    def _merge_results(self, text_results, image_results, top_k) -> list[dict]:
        seen = {}
        for r in text_results + image_results:
            if r['id'] not in seen or r['similarity_score'] > seen[r['id']]['similarity_score']:
                seen[r['id']] = r
        merged = sorted(seen.values(), key=lambda x: x['similarity_score'], reverse=True)
        return merged[:top_k]

    @property
    def is_loaded(self) -> bool:
        return all([
            self.client is not None,
            self.img_collection is not None,
            self.txt_collection is not None,
            self.text_model is not None,
            self.clip_model is not None,
        ])