"""Answer evaluation checklist helper for manual groundedness review."""

import json
from pathlib import Path


def main() -> None:
    questions_path = Path(__file__).parent / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    present = sum(item["answer_present"] for item in questions)
    print(
        f"Evaluation set: {len(questions)} questions "
        f"({present} answerable, {len(questions) - present} unanswerable)."
    )
    print(
        "Review each answer for correctness, groundedness, language match, "
        "and source citation."
    )


if __name__ == "__main__":
    main()
