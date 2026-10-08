import re


DISCLAIMER = (
    "This application is an AI-assisted educational/research "
    "tool and is not a medical diagnostic or treatment system. "
    "Its outputs should not be used as a substitute for advice "
    "from a qualified healthcare professional."
)


DIAGNOSIS_PATTERNS = [
    r"\byou have\b",
    r"\byou definitely have\b",
    r"\bthis confirms\b",
    r"\bconfirmed diagnosis\b",
    r"\bdiagnosis ispython -m pip install sentence-transformers\b",
    r"\byou are dipython -m pip install sentence-transformersagnosed\b",
]


MEDICATION_PATTERNS = [
    r"\btake \d+",
    r"\btake this medication\b",
    r"\bprescribe\b",
    r"\bprescription\b",
    r"\bdosage\b",
    r"\bmg per day\b",
    r"\bstop taking\b",
    r"\bstart taking\b",
]


DANGEROUS_PATTERNS = [
    r"\bperform surgery\b",
    r"\btreat yourself\b",
    r"\bignore your doctor\b",
    r"\bavoid medical care\b",
    r"\bdo not seek medical\b",
]


CERTAINTY_PATTERNS = [
    r"\b100% certain\b",
    r"\bcompletely certain\b",
    r"\bdefinitely\b",
    r"\bwithout doubt\b",
    r"\bguaranteed\b",
]


def contains_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    """
    Return True if any safety pattern is detected.
    """

    text = text.lower()

    for pattern in patterns:
        if re.search(
            pattern,
            text,
        ):
            return True

    return False


def check_response(
    text: str,
) -> dict:
    """
    Check an AI-generated response for simple
    healthcare safety problems.
    """

    if not text or not text.strip():
        return {
            "status": "warning",
            "message": (
                "No response was generated."
            ),
        }

    if contains_pattern(
        text,
        DANGEROUS_PATTERNS,
    ):
        return {
            "status": "blocked",
            "message": (
                "The generated response contained potentially "
                "dangerous medical instructions and was blocked."
            ),
        }

    if contains_pattern(
        text,
        MEDICATION_PATTERNS,
    ):
        return {
            "status": "blocked",
            "message": (
                "The generated response contained medication "
                "or prescription-style instructions and was blocked."
            ),
        }

    if contains_pattern(
        text,
        DIAGNOSIS_PATTERNS,
    ):
        return {
            "status": "warning",
            "message": (
                "The generated response contained language "
                "that could be interpreted as a definitive diagnosis."
            ),
        }

    if contains_pattern(
        text,
        CERTAINTY_PATTERNS,
    ):
        return {
            "status": "warning",
            "message": (
                "The generated response contained unsupported "
                "certainty language."
            ),
        }

    return {
        "status": "safe",
        "message": (
            "No basic safety-pattern violation was detected."
        ),
    }


def add_disclaimer(
    text: str,
) -> str:
    """
    Always attach the healthcare disclaimer.
    """

    return (
        text.strip()
        + "\n\n"
        + "Safety Notice:\n"
        + DISCLAIMER
    )


def sanitize_response(
    text: str,
    safety_result: dict,
) -> str:
    """
    Modify or block an unsafe generated response.
    """

    if safety_result["status"] == "blocked":
        safe_text = (
            "The AI-generated summary was not displayed "
            "because it contained language that could be "
            "unsafe or clinically overconfident."
        )

        return add_disclaimer(
            safe_text
        )

    if safety_result["status"] == "warning":
        safe_text = (
            "The generated explanation requires caution "
            "because it contained language that may imply "
            "more certainty than the available evidence supports."
            "\n\n"
            + text
        )

        return add_disclaimer(
            safe_text
        )

    return add_disclaimer(
        text
    )


if __name__ == "__main__":

    test_text = (
        "You definitely have cancer. "
        "Take this medication immediately."
    )

    result = check_response(
        test_text
    )

    print(result)