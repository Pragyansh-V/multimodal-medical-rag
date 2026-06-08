# scripts/test_gemini.py
# Quick sanity check — verifies Gemini API key and vision capability

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
import urllib.request
import io

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY not found in .env")
    sys.exit(1)

# Configure
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

print("✅ Gemini API configured")
print(f"   Model: {MODEL}")

# ── Test 1: Text only ─────────────────────────────────────────────────────────
print("\n── Test 1: Text query ──")
response = client.models.generate_content(
    model=MODEL,
    contents="What is a pathology slide? Answer in one sentence."
)
print(f"   Response: {response.text.strip()}")

# ── Test 2: Vision — create a simple test image ───────────────────────────────
print("\n── Test 2: Vision query ──")

try:
    import io
    # Create a simple synthetic image to test vision capability
    img = Image.new("RGB", (224, 224), color=(180, 120, 100))
    
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            "Describe the dominant color in this image in one sentence."
        ]
    )
    print(f"   Response: {response.text.strip()}")
    print("\n✅ Vision capability confirmed — Gemini can process images")

except Exception as e:
    print(f"❌ Vision test failed: {e}")
    sys.exit(1)