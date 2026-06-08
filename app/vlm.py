# app/vlm.py

import os
import io
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL          = "gemini-2.5-flash"
MAX_IMAGES     = 3  # max images to send per request — keeps tokens manageable


class VLMClient:
    """
    Wraps Gemini 2.5 Flash for multimodal medical Q&A.
    Takes a question + retrieved contexts → returns an answer.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model  = MODEL
        print(f"✅ VLM client initialised — {self.model}")

    def answer(
        self,
        question: str,
        retrieved_contexts: list[dict],
    ) -> str:
        """
        Generate an answer using Gemini with retrieved pathology
        images and Q&A pairs as context.

        Args:
            question           : user's clinical question
            retrieved_contexts : list of dicts from retriever

        Returns:
            answer string from Gemini
        """

        # ── Build prompt ──────────────────────────────────────────────────────
        context_text = self._build_context_text(retrieved_contexts)
        prompt       = self._build_prompt(question, context_text)

        # ── Load images ───────────────────────────────────────────────────────
        image_parts = self._load_images(retrieved_contexts)

        # ── Build content list ────────────────────────────────────────────────
        # Structure: [image1, image2, image3, prompt_text]
        contents = image_parts + [prompt]

        # ── Call Gemini ───────────────────────────────────────────────────────
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert medical AI assistant specialising in pathology. "
                    "You are given pathology images and related Q&A pairs as context. "
                    "Answer the question accurately and concisely based on the visual "
                    "evidence and context provided. If you are uncertain, say so clearly."
                ),
                temperature=0.1,  # low temperature — medical answers need precision
                max_output_tokens=512,
            )
        )

        return response.text.strip()

    def _build_context_text(self, contexts: list[dict]) -> str:
        """Format retrieved Q&A pairs as readable context."""
        lines = ["--- Retrieved Medical Context ---"]
        for i, ctx in enumerate(contexts, 1):
            lines.append(
                f"\n[Context {i}] (similarity: {ctx['similarity_score']:.3f})"
                f"\n  Q: {ctx['question']}"
                f"\n  A: {ctx['answer']}"
            )
        return "\n".join(lines)

    def _build_prompt(self, question: str, context_text: str) -> str:
        """Build the final prompt combining question and context."""
        return (
            f"{context_text}\n\n"
            f"--- Question ---\n"
            f"{question}\n\n"
            f"Based on the pathology images shown and the retrieved context above, "
            f"please provide a precise medical answer."
        )

    def _load_images(self, contexts: list[dict]) -> list:
        """Load images from disk and convert to Gemini Parts."""
        parts  = []
        loaded = 0

        for ctx in contexts[:MAX_IMAGES]:
            img_path = Path(ctx['image_path'])
            if not img_path.exists():
                print(f"  ⚠️  Image not found: {img_path}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                img_bytes = buf.getvalue()

                parts.append(
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg"
                    )
                )
                loaded += 1

            except Exception as e:
                print(f"  ⚠️  Failed to load image {img_path}: {e}")

        print(f"  Loaded {loaded} images for VLM context")
        return parts

    @property
    def is_available(self) -> bool:
        return self.client is not None