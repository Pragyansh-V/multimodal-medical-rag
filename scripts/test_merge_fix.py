# scripts/test_merge_fix.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retriever import MultimodalRetriever

retriever = MultimodalRetriever()

question = "is there evidence of inflammation?"
results = retriever.retrieve(question=question, top_k=3, mode="hybrid")

print(f"\n{'='*60}")
print(f"  Hybrid retrieval results for: '{question}'")
print(f"{'='*60}\n")

for r in results:
    print(f"  id: {r['id']:15s}  score: {r['similarity_score']:.4f}  answer: {r['answer']}")

text_ids  = {"pathvqa_42", "pathvqa_148", "pathvqa_235"}
image_ids = {"pathvqa_485", "pathvqa_442", "pathvqa_355"}

returned_ids = {r['id'] for r in results}
from_text  = returned_ids & text_ids
from_image = returned_ids & image_ids

print(f"\n  From text-mode results : {from_text or 'none'}")
print(f"  From image-mode results: {from_image or 'none'}")
print(f"\n  {'✅ Mix of both — fix working' if from_text and from_image else '⚠️  Still all from one mode — fix may not be working'}")