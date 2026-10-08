# Evidence-Aware Multimodal Healthcare AI — MVP

## Overview

This project is a small educational/research MVP demonstrating an
evidence-aware multimodal healthcare AI assistant.

The application combines:

1. Structured machine learning
2. Medical image classification
3. Retrieval-Augmented Generation (RAG)
4. Rule-based evidence fusion
5. Simple confidence/uncertainty interpretation
6. Basic safety guardrails
7. Evidence-grounded LLM summaries

The project is intentionally designed to remain small and understandable.

---

# Important Disclaimer

This application is NOT a medical diagnostic system.

It does not provide clinical diagnosis, treatment recommendations,
prescriptions, or professional medical advice.

The machine-learning and computer-vision outputs are model predictions only.

Model confidence is not the same as clinical certainty.

The application is intended for learning, portfolio demonstration,
and academic/research purposes.

---

# Core Architecture

```text
                 User
                  |
        +---------+---------+
        |         |         |
   Structured   Image    Question
      Data       |          |
        |        |          |
        v        v          v
   +---------+ +---------+ +---------+
   | Logistic| | ResNet18| |   RAG   |
   |Regression| | Vision | | + FAISS |
   +----+----+ +----+----+ +----+----+
        |           |           |
        +-----------+-----------+
                    |
                    v
             Evidence Fusion
                    |
                    v
          Confidence / Uncertainty
                    |
                    v
                Safety
                    |
                    v
             LLM Summary
                    |
                    v
               Streamlit