import os
import sys
from pathlib import Path

import faiss
import joblib
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]

DOCUMENT_DIR = ROOT_DIR / "data" / "documents"

INDEX_PATH = ROOT_DIR / "models" / "rag.index"
METADATA_PATH = ROOT_DIR / "models" / "rag_metadata.pkl"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 4

load_dotenv(
    ROOT_DIR / ".env"
)


def load_documents() -> list[dict]:
    """
    Read all PDF documents from data/documents/.
    """

    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENT_DIR}"
        )

    pdf_files = list(
        DOCUMENT_DIR.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {DOCUMENT_DIR}"
        )

    documents = []

    for pdf_path in pdf_files:

        reader = PdfReader(
            str(pdf_path)
        )

        pages = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages.append(text)

        full_text = "\n".join(pages).strip()

        if full_text:
            documents.append(
                {
                    "source": pdf_path.name,
                    "text": full_text,
                }
            )

    if not documents:
        raise ValueError(
            "PDF files were found, but no readable text "
            "could be extracted from them."
        )

    return documents


def create_chunks(
    documents: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Split documents into overlapping text chunks.
    """

    chunks = []

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source": document["source"],
                        "text": chunk_text,
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def get_embedding_model():
    """
    Load the Sentence Transformer embedding model.
    """

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


def build_vector_store():
    """
    Build a FAISS vector index from local PDFs.
    """

    documents = load_documents()

    chunks = create_chunks(
        documents
    )

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    embedding_model = (
        get_embedding_model()
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    joblib.dump(
        chunks,
        METADATA_PATH,
    )

    print(
        f"Created FAISS index: {INDEX_PATH}"
    )

    print(
        f"Stored chunks: {len(chunks)}"
    )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
    }


def load_vector_store():
    """
    Load the FAISS index and chunk metadata.
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run:\n"
            "python modules/rag.py --build"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"RAG metadata not found at {METADATA_PATH}."
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    chunks = joblib.load(
        METADATA_PATH
    )

    embedding_model = (
        get_embedding_model()
    )

    return (
        index,
        chunks,
        embedding_model,
    )


def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant evidence chunks.
    """

    if not query or not query.strip():
        raise ValueError(
            "A question is required."
        )

    index, chunks, embedding_model = (
        load_vector_store()
    )

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):

        if index_position < 0:
            continue

        if index_position >= len(chunks):
            continue

        chunk = chunks[
            index_position
        ]

        results.append(
            {
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score),
            }
        )

    return results


def _get_openai_client():
    """
    Create the OpenAI client from the environment variable.
    """

    api_key = os.getenv(
        "LLM_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY is missing. "
            "Add it to the project's .env file."
        )

    return OpenAI(
        api_key=api_key
    )


def generate_answer(
    query: str,
    retrieved_documents: list[dict],
) -> str:
    """
    Generate an evidence-grounded answer.

    The model is explicitly instructed to use only
    the retrieved evidence.
    """

    if not retrieved_documents:
        return (
            "No relevant evidence was retrieved. "
            "The system cannot provide an evidence-grounded answer."
        )

    client = _get_openai_client()

    evidence_blocks = []

    for number, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        evidence_blocks.append(
            f"""
SOURCE {number}: {document["source"]}

EVIDENCE:
{document["text"]}
"""
        )

    evidence = "\n".join(
        evidence_blocks
    )

    instructions = """
You are an evidence-grounded educational healthcare assistant.

Use ONLY the evidence supplied by the retrieval system.

Do not:
- diagnose the user
- prescribe medication
- provide treatment instructions
- invent facts
- invent citations
- claim certainty
- pretend that a model prediction is a clinical diagnosis

Clearly distinguish:
1. Information stated in the retrieved evidence.
2. Information that cannot be established from the evidence.
3. The need for professional medical evaluation when appropriate.

Answer briefly and clearly.
"""

    user_input = f"""
Healthcare question:

{query}

Retrieved evidence:

{evidence}

Answer the question using only the retrieved evidence.
Mention the source filenames when referring to the evidence.
Do not create sources that are not present above.
"""

    response = client.responses.create(
        model="gpt-6-luna",
        instructions=instructions,
        input=user_input,
    )

    return response.output_text.strip()


def generate_final_summary(
    risk_result: dict | None,
    vision_result: dict | None,
    fusion_result: dict,
    rag_answer: str | None,
    retrieved_documents: list[dict],
) -> str:
    """
    Generate the final evidence-aware summary.

    The LLM receives model results plus retrieved evidence.
    """

    client = _get_openai_client()

    risk_text = (
        str(risk_result)
        if risk_result
        else "No structured model result available."
    )

    vision_text = (
        str(vision_result)
        if vision_result
        else "No image model result available."
    )

    fusion_text = str(
        fusion_result
    )

    evidence_text = "\n".join(
        [
            f"{item['source']}: {item['text']}"
            for item in retrieved_documents
        ]
    )

    if not evidence_text:
        evidence_text = (
            "No retrieved medical evidence was available."
        )

    instructions = """
You are generating a short educational summary for an
evidence-aware healthcare AI demonstration.

Important rules:

- Do not provide a medical diagnosis.
- Do not prescribe medication.
- Do not recommend a treatment.
- Do not invent medical evidence.
- Do not invent sources.
- Do not treat model probabilities as clinical certainty.
- Clearly distinguish model predictions from retrieved evidence.
- Explicitly mention uncertainty or disagreement.
- If signals conflict, say that they conflict.
- If evidence is insufficient, say so.
- Keep the response concise.
"""

    user_input = f"""
STRUCTURED MODEL RESULT:
{risk_text}

VISION MODEL RESULT:
{vision_text}

EVIDENCE FUSION:
{fusion_text}

RAG ANSWER:
{rag_answer or "No RAG answer was generated."}

RETRIEVED EVIDENCE:
{evidence_text}

Write a short human-readable summary.
"""

    response = client.responses.create(
        model="gpt-6-luna",
        instructions=instructions,
        input=user_input,
    )

    return response.output_text.strip()


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_vector_store()
    else:
        print(
            "Usage:"
        )
        print(
            "python modules/rag.py --build"
        )