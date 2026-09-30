import re


class TextNormalizer:

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize common PDF extraction artifacts
        without modifying ordinary words containing 'I'.
        """

        # PyMuPDF may extract the ₹ symbol as the
        # capital letter 'I' when followed by a number.
        text = re.sub(
            r"\bI(?=\d)",
            "₹",
            text,
        )

        return text