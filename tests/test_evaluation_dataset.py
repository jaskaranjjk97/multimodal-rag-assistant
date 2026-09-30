from evaluation.dataset import EVALUATION_CASES


def test_evaluation_dataset_contains_cases():

    assert len(EVALUATION_CASES) == 3


def test_evaluation_cases_have_required_fields():

    for case in EVALUATION_CASES:

        assert case.question
        assert case.expected_answer


def test_revenue_2024_case_exists():

    questions = [
        case.question
        for case in EVALUATION_CASES
    ]

    assert (
        "What was Acme Technologies' revenue in 2024?"
        in questions
    )