from dataclasses import dataclass


@dataclass
class EvaluationCase:
    question: str
    expected_answer: str

EVALUATION_CASES = [
    EvaluationCase(
        question="What was Acme Technologies' revenue in 2024?",
        expected_answer="₹10 crore",
    ),
    EvaluationCase(
        question="Which product had the highest growth rate?",
        expected_answer="Automation, 32%",
    ),
    EvaluationCase(
        question=(
            "What was Acme Technologies' annual revenue trend "
            "from 2022 to 2025?"
        ),
        expected_answer=(
            "₹6 crore in 2022, ₹8 crore in 2023, "
            "₹10 crore in 2024, and ₹12 crore in 2025."
        ),
    ),
]