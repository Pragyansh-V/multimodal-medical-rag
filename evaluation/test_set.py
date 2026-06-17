# evaluation/test_set.py
# Ground-truth test questions for RAGAS evaluation
# Pulled directly from PathVQA metadata — these have known correct answers

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"


def load_test_set(n: int = 10, answer_type: str = "open_ended") -> list[dict]:
    """
    Loads a small evaluation test set from the existing PathVQA metadata.

    Args:
        n           : number of test questions to use
        answer_type : 'open_ended', 'yes_no', or 'all'

    Returns:
        list of dicts with 'question' and 'ground_truth' keys
    """
    with open(METADATA_PATH) as f:
        all_data = json.load(f)

    if answer_type != "all":
        filtered = [d for d in all_data if d['answer_type'] == answer_type]
    else:
        filtered = all_data

    # Take first n for reproducibility — no random sampling needed here
    selected = filtered[:n]

    test_set = [
        {
            "question": item['question'],
            "ground_truth": item['answer'],
            "id": item['id']
        }
        for item in selected
    ]

    return test_set


if __name__ == "__main__":
    ts = load_test_set(n=5)
    for item in ts:
        print(f"Q: {item['question']}")
        print(f"A: {item['ground_truth']}")
        print()