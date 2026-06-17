# evaluation/ragas_eval.py

import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "app"))

load_dotenv(PROJECT_ROOT / ".env")

from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.retriever import MultimodalRetriever
from app.vlm import VLMClient
from evaluation.test_set import load_test_set


def build_ragas_dataset(test_questions: list[dict], retriever, vlm) -> EvaluationDataset:
    """
    Runs each test question through the actual RAG pipeline,
    collects question, answer, retrieved_contexts, and ground_truth
    in the format RAGAS expects.
    """
    samples = []

    for item in test_questions:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  Running: {question[:60]}...")

        # Run through actual pipeline — text mode for speed/consistency
        retrieved = retriever.retrieve(question=question, top_k=3, mode="hybrid")
        answer = vlm.answer(question=question, retrieved_contexts=retrieved)

        # RAGAS needs contexts as plain strings
        context_strings = [
            f"Q: {ctx['question']} A: {ctx['answer']}"
            for ctx in retrieved
        ]

        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": context_strings,
            "reference": ground_truth,
        })

        #time.sleep(4)  # avoid rate limits

    return EvaluationDataset.from_list(samples)


def run_evaluation(n_questions: int = 5):
    """
    Full RAGAS evaluation pipeline:
    1. Load test questions
    2. Run through actual RAG pipeline
    3. Score with 4 RAGAS metrics
    4. Print report
    """

    print(f"\n{'═'*60}")
    print(f"  RAGAS Evaluation — Multimodal Medical RAG")
    print(f"{'═'*60}\n")

    # ── Load components ──────────────────────────────────────────────────────
    print("Loading retriever...")
    retriever = MultimodalRetriever()

    print("Loading VLM...")
    vlm = VLMClient()

    # ── Configure RAGAS judge LLM ────────────────────────────────────────────
    print("Configuring RAGAS judge (Gemini)...")
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="meta/llama-3.1-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    )

    # ── Load test set ─────────────────────────────────────────────────────────
    print(f"Loading {n_questions} test questions...")
    test_questions = load_test_set(n=n_questions, answer_type="open_ended")

    # ── Run pipeline on each question ────────────────────────────────────────
    print("\nRunning RAG pipeline on test questions:")
    dataset = build_ragas_dataset(test_questions, retriever, vlm)

    # ── Score with RAGAS ──────────────────────────────────────────────────────
    print("\nScoring with RAGAS metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(timeout=180, max_workers=1, max_retries=5),
    )

    # ── Report ────────────────────────────────────────────────────────────────
    df = results.to_pandas()

    print(f"\n{'═'*60}")
    print(f"  RAGAS Scores (averaged across {len(df)} questions)")
    print(f"{'═'*60}")
    print(f"  Faithfulness      : {df['faithfulness'].mean():.4f}")
    print(f"  Answer Relevancy  : {df['answer_relevancy'].mean():.4f}")
    print(f"  Context Precision : {df['context_precision'].mean():.4f}")
    print(f"  Context Recall    : {df['context_recall'].mean():.4f}")
    print(f"{'═'*60}\n")

    # Save full results
    output_path = PROJECT_ROOT / "evaluation" / "ragas_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Full results saved to: {output_path}")

    return df


if __name__ == "__main__":
    run_evaluation(n_questions=3)