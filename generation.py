"""
Answer generation using a local Ollama model, given a user
question and the retrieved context chunks from the vector store.
"""

import os

import ollama

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context excerpts from uploaded PDF documents. "
    "If the answer cannot be found in the context, say so clearly instead "
    "of making something up. Always be concise and cite which source file "
    "(by name) the information came from when relevant."
)


def build_context_block(retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "No relevant context was found."

    parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        parts.append(
            f"[Excerpt {i} — source: {chunk['source_file']}, "
            f"chunk #{chunk['chunk_index']}, relevance score: {chunk['score']:.3f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def generate_answer_stream(question: str, retrieved_chunks: list[dict]):
    """
    Streams the generated answer token by token from Ollama.
    Yields string chunks as they arrive.
    """
    context_block = build_context_block(retrieved_chunks)

    user_prompt = (
        f"Context excerpts from the uploaded documents:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}\n\n"
        "Answer the question using only the context above."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        client = ollama.Client(host=OLLAMA_HOST)
        for chunk in client.chat(model=MODEL_NAME, messages=messages, stream=True):
            token = chunk["message"]["content"]
            if token:
                yield token
    except Exception as exc:
        yield f"Sorry, the local model could not be reached: {exc}"


def get_model_status() -> tuple[bool, str]:
    """Return whether the configured local Ollama model is available."""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.list()

        if hasattr(response, "models"):
            models = response.models
        elif isinstance(response, dict):
            models = response.get("models", [])
        elif isinstance(response, list):
            models = response
        else:
            models = []

        names = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model") or ""
            else:
                name = getattr(model, "name", None) or getattr(model, "model", None) or ""
            if name:
                names.append(name)

        if MODEL_NAME in names:
            return True, f"{MODEL_NAME} is ready"
        if any(name.startswith(MODEL_NAME.split(":")[0]) for name in names):
            return True, f"{MODEL_NAME} is ready"
        return False, f"{MODEL_NAME} was not found in the local Ollama models list"
    except Exception as exc:
        return False, str(exc)


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """Non-streaming convenience wrapper, returns the full answer string."""
    return "".join(generate_answer_stream(question, retrieved_chunks))
