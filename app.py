from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from modules.risk_model import (
    FEATURE_NAMES,
    get_feature_importance,
    predict_risk,
)

from modules.vision_model import (
    predict_image,
)

from modules.rag import (
    retrieve_documents,
    generate_answer,
    generate_final_summary,
)

from modules.fusion import (
    fuse_evidence,
)

from modules.safety import (
    DISCLAIMER,
    check_response,
    sanitize_response,
)


ROOT_DIR = Path(__file__).resolve().parent

load_dotenv(
    ROOT_DIR / ".env"
)


st.set_page_config(
    page_title="Evidence-Aware Healthcare AI",
    page_icon="🩺",
    layout="wide",
)


st.title(
    "🩺 Evidence-Aware Multimodal Healthcare AI"
)

st.caption(
    "Educational / research MVP — not a clinical diagnostic system"
)

st.warning(
    DISCLAIMER
)


# ---------------------------------------------------------
# Patient / structured information
# ---------------------------------------------------------

st.header(
    "1. Structured Patient / Cell Measurements"
)

st.info(
    "This MVP uses the UCI Breast Cancer Wisconsin Diagnostic "
    "dataset. The structured inputs are numeric cell-nucleus "
    "measurements rather than general demographic information."
)

column1, column2 = st.columns(2)

with column1:

    mean_radius = st.number_input(
        "Mean Radius",
        min_value=0.0,
        value=14.0,
        step=0.1,
    )

    mean_texture = st.number_input(
        "Mean Texture",
        min_value=0.0,
        value=19.0,
        step=0.1,
    )

    mean_perimeter = st.number_input(
        "Mean Perimeter",
        min_value=0.0,
        value=90.0,
        step=0.1,
    )

    mean_area = st.number_input(
        "Mean Area",
        min_value=0.0,
        value=600.0,
        step=1.0,
    )

    mean_smoothness = st.number_input(
        "Mean Smoothness",
        min_value=0.0,
        value=0.1,
        step=0.001,
        format="%.4f",
    )

with column2:

    mean_compactness = st.number_input(
        "Mean Compactness",
        min_value=0.0,
        value=0.1,
        step=0.001,
        format="%.4f",
    )

    mean_concavity = st.number_input(
        "Mean Concavity",
        min_value=0.0,
        value=0.1,
        step=0.001,
        format="%.4f",
    )

    mean_concave_points = st.number_input(
        "Mean Concave Points",
        min_value=0.0,
        value=0.05,
        step=0.001,
        format="%.4f",
    )

    mean_symmetry = st.number_input(
        "Mean Symmetry",
        min_value=0.0,
        value=0.18,
        step=0.001,
        format="%.4f",
    )

    mean_fractal_dimension = st.number_input(
        "Mean Fractal Dimension",
        min_value=0.0,
        value=0.06,
        step=0.001,
        format="%.4f",
    )


patient_data = {
    "mean_radius": mean_radius,
    "mean_texture": mean_texture,
    "mean_perimeter": mean_perimeter,
    "mean_area": mean_area,
    "mean_smoothness": mean_smoothness,
    "mean_compactness": mean_compactness,
    "mean_concavity": mean_concavity,
    "mean_concave_points": mean_concave_points,
    "mean_symmetry": mean_symmetry,
    "mean_fractal_dimension": mean_fractal_dimension,
}


# ---------------------------------------------------------
# Clinical notes
# ---------------------------------------------------------

st.header(
    "2. Symptoms / Clinical Notes"
)

clinical_notes = st.text_area(
    "Enter symptoms or clinical notes",
    placeholder=(
        "Example: The patient reports a breast lump "
        "noticed recently..."
    ),
    height=120,
)


# ---------------------------------------------------------
# Medical image
# ---------------------------------------------------------

st.header(
    "3. Medical Image"
)

uploaded_image = st.file_uploader(
    "Upload a breast ultrasound image",
    type=[
        "png",
        "jpg",
        "jpeg",
    ],
)


if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Uploaded image",
        width=400,
    )


# ---------------------------------------------------------
# Healthcare question
# ---------------------------------------------------------

st.header(
    "4. Healthcare Question"
)

healthcare_question = st.text_area(
    "Ask a healthcare-related question",
    placeholder=(
        "Example: What are common signs and diagnostic "
        "approaches for breast cancer?"
    ),
    height=120,
)


# ---------------------------------------------------------
# Analysis
# ---------------------------------------------------------

analyze_button = st.button(
    "🔎 Analyze Case",
    type="primary",
)


if analyze_button:

    risk_result = None
    vision_result = None
    rag_answer = None
    retrieved_documents = []
    fusion_result = None
    final_summary = None

    # -----------------------------------------------------
    # Structured ML
    # -----------------------------------------------------

    st.header(
        "5. Structured ML Result"
    )

    try:

        risk_result = predict_risk(
            patient_data
        )

        st.success(
            "Structured model prediction generated."
        )

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        with result_col1:
            st.metric(
                "Prediction",
                risk_result["prediction"],
            )

        with result_col2:
            st.metric(
                "Probability",
                f"{risk_result['probability'] * 100:.2f}%",
            )

        with result_col3:
            st.metric(
                "Model Confidence",
                risk_result["confidence"],
            )

        st.caption(
            "This is a model prediction, not a medical diagnosis."
        )

        try:

            importance = get_feature_importance()

            st.subheader(
                "Important Structured Features"
            )

            for item in importance[:5]:
                st.write(
                    f"**{item['feature']}** — "
                    f"coefficient magnitude: "
                    f"{item['importance']:.4f}"
                )

        except Exception as error:

            st.warning(
                f"Feature importance could not be displayed: {error}"
            )

    except Exception as error:

        st.error(
            f"Structured model error: {error}"
        )


    # -----------------------------------------------------
    # Vision
    # -----------------------------------------------------

    st.header(
        "6. Medical Image Model"
    )

    if uploaded_image is None:

        st.warning(
            "No medical image was uploaded. "
            "The vision model was skipped."
        )

    else:

        try:

            vision_result = predict_image(
                uploaded_image
            )

            st.success(
                "Image model prediction generated."
            )

            vision_col1, vision_col2 = (
                st.columns(2)
            )

            with vision_col1:
                st.metric(
                    "Predicted Class",
                    vision_result[
                        "predicted_class"
                    ],
                )

            with vision_col2:
                st.metric(
                    "Model Confidence",
                    f"{vision_result['confidence'] * 100:.2f}%",
                )

            st.caption(
                "This is an image-classification model prediction, "
                "not a medical diagnosis."
            )

        except Exception as error:

            st.error(
                f"Vision model error: {error}"
            )


    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    st.header(
        "7. Retrieved Medical Evidence"
    )

    if not healthcare_question.strip():

        st.warning(
            "No healthcare question was entered. "
            "RAG retrieval was skipped."
        )

    else:

        try:

            retrieved_documents = (
                retrieve_documents(
                    healthcare_question
                )
            )

            if not retrieved_documents:

                st.warning(
                    "No relevant evidence was retrieved."
                )

            else:

                st.subheader(
                    "Retrieved Evidence"
                )

                for number, document in enumerate(
                    retrieved_documents,
                    start=1,
                ):

                    with st.expander(
                        f"Source {number}: {document['source']}"
                    ):

                        st.write(
                            f"Similarity score: "
                            f"{document['score']:.4f}"
                        )

                        st.write(
                            document["text"]
                        )

                try:

                    rag_answer = generate_answer(
                        healthcare_question,
                        retrieved_documents,
                    )

                    st.subheader(
                        "RAG Answer"
                    )

                    st.write(
                        rag_answer
                    )

                except Exception as error:

                    st.error(
                        f"LLM/RAG answer error: {error}"
                    )

        except Exception as error:

            st.error(
                f"RAG retrieval error: {error}"
            )


    # -----------------------------------------------------
    # Fusion
    # -----------------------------------------------------

    st.header(
        "8. Evidence Fusion"
    )

    try:

        rag_result = {
            "retrieved": len(
                retrieved_documents
            ) > 0
        }

        fusion_result = fuse_evidence(
            risk_result,
            vision_result,
            rag_result,
        )

        fusion_col1, fusion_col2 = (
            st.columns(2)
        )

        with fusion_col1:

            st.metric(
                "Evidence Status",
                fusion_result["status"],
            )

        with fusion_col2:

            st.metric(
                "Uncertainty",
                fusion_result["uncertainty"],
            )

        st.write(
            fusion_result["explanation"]
        )

        st.caption(
            "The fusion result is a transparent rule-based "
            "comparison of model signals. It is not a clinical "
            "decision."
        )

    except Exception as error:

        st.error(
            f"Fusion error: {error}"
        )


    # -----------------------------------------------------
    # Final AI summary
    # -----------------------------------------------------

    st.header(
        "9. Final AI Summary"
    )

    if fusion_result is None:

        st.warning(
            "The final summary could not be generated because "
            "evidence fusion did not complete."
        )

    else:

        try:

            final_summary = generate_final_summary(
                risk_result=risk_result,
                vision_result=vision_result,
                fusion_result=fusion_result,
                rag_answer=rag_answer,
                retrieved_documents=retrieved_documents,
            )

            safety_result = check_response(
                final_summary
            )

            safe_summary = sanitize_response(
                final_summary,
                safety_result,
            )

            st.write(
                safe_summary
            )

            st.subheader(
                "Safety Status"
            )

            if safety_result["status"] == "safe":

                st.success(
                    safety_result["message"]
                )

            elif safety_result["status"] == "warning":

                st.warning(
                    safety_result["message"]
                )

            else:

                st.error(
                    safety_result["message"]
                )

        except Exception as error:

            st.error(
                f"Final AI summary error: {error}"
            )


    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    st.header(
        "10. Sources Used"
    )

    if retrieved_documents:

        unique_sources = []

        for document in retrieved_documents:

            source = document["source"]

            if source not in unique_sources:
                unique_sources.append(
                    source
                )

        for number, source in enumerate(
            unique_sources,
            start=1,
        ):

            st.write(
                f"{number}. {source}"
            )

    else:

        st.info(
            "No RAG sources were retrieved."
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Evidence-Aware Multimodal Healthcare AI — MVP"
)

st.caption(
    DISCLAIMER
)