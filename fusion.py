def confidence_level(value: float) -> str:
    """
    Convert probability/confidence into a simple category.
    """

    if value < 0.60:
        return "Low"

    if value < 0.80:
        return "Moderate"

    return "Higher"


def calculate_uncertainty(
    risk_result: dict | None,
    vision_result: dict | None,
    status: str,
) -> str:
    """
    Calculate simple uncertainty.

    This is not a clinical uncertainty calculation.
    """

    if status == "Conflict":
        return "High"

    if status == "Insufficient Evidence":
        return "High"

    confidence_values = []

    if risk_result:
        confidence_values.append(
            max(
                risk_result.get(
                    "probability",
                    0.5,
                ),
                1
                - risk_result.get(
                    "probability",
                    0.5,
                ),
            )
        )

    if vision_result:
        confidence_values.append(
            vision_result.get(
                "confidence",
                0.5,
            )
        )

    if not confidence_values:
        return "High"

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    if average_confidence < 0.60:
        return "High"

    if average_confidence < 0.80:
        return "Moderate"

    return "Lower"


def fuse_evidence(
    risk_result: dict | None,
    vision_result: dict | None,
    rag_result: dict | None,
) -> dict:
    """
    Rule-based multimodal evidence fusion.

    The structured and vision models both operate in the
    breast-cancer domain in this MVP.
    """

    if risk_result is None and vision_result is None:
        return {
            "status": "Insufficient Evidence",
            "uncertainty": "High",
            "explanation": (
                "No structured or image model result "
                "was available."
            ),
        }

    if risk_result is None or vision_result is None:
        status = "Insufficient Evidence"

        return {
            "status": status,
            "uncertainty": "High",
            "explanation": (
                "Only one model signal was available. "
                "A multimodal comparison could not be completed."
            ),
        }

    risk_class = risk_result.get(
        "class"
    )

    vision_class = vision_result.get(
        "predicted_class",
        ""
    ).lower()

    if (
        risk_class == "malignant"
        and vision_class == "malignant"
    ):
        status = "Agreement"

        explanation = (
            "The structured model and image model both "
            "produced malignant-pattern predictions. "
            "This is agreement between model outputs, "
            "not a clinical diagnosis."
        )

    elif (
        risk_class == "benign"
        and vision_class in {
            "benign",
            "normal",
        }
    ):
        status = "Partial Agreement"

        explanation = (
            "The structured model produced a benign-pattern "
            "prediction and the image model produced either "
            "a benign or normal prediction."
        )

    elif (
        risk_class == "malignant"
        and vision_class in {
            "benign",
            "normal",
        }
    ):
        status = "Conflict"

        explanation = (
            "The structured model and image model produced "
            "different signals. The disagreement increases "
            "uncertainty."
        )

    elif (
        risk_class == "benign"
        and vision_class == "malignant"
    ):
        status = "Conflict"

        explanation = (
            "The structured model produced a benign-pattern "
            "prediction while the image model produced a "
            "malignant prediction."
        )

    else:
        status = "Partial Agreement"

        explanation = (
            "The available model outputs are not directly "
            "identical, but they provide partially compatible "
            "signals."
        )

    uncertainty = calculate_uncertainty(
        risk_result,
        vision_result,
        status,
    )

    if rag_result:
        explanation += (
            " Retrieved medical evidence is available "
            "for contextual support."
        )
    else:
        explanation += (
            " No retrieved medical evidence was available."
        )

    return {
        "status": status,
        "uncertainty": uncertainty,
        "explanation": explanation,
    }