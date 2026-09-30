from ingestion.extractors.text_normalizer import TextNormalizer


def test_normalizes_rupee_extraction_artifact():

    text = (
        "Revenue increased from I6 crore "
        "to I12 crore."
    )

    normalized = TextNormalizer.normalize(text)

    assert normalized == (
        "Revenue increased from ₹6 crore "
        "to ₹12 crore."
    )


def test_normalizes_rupee_with_cr():

    text = "AI Assistant generated I5 Cr."

    normalized = TextNormalizer.normalize(text)

    assert normalized == (
        "AI Assistant generated ₹5 Cr."
    )


def test_does_not_modify_normal_words():

    text = (
        "India provides Information about "
        "AI Assistant."
    )

    normalized = TextNormalizer.normalize(text)

    assert normalized == text


def test_normalizes_multiple_values():

    text = "I6, I8, I10, I12"

    normalized = TextNormalizer.normalize(text)

    assert normalized == "₹6, ₹8, ₹10, ₹12"