from backend.context_builder import format_context
from backend.llm import generate
from backend.retrieval import retrieve


class RAGUnavailableError(RuntimeError):
    """Raised when the codebase index cannot answer requests."""


def ask_codebase(question, project_id=None):
    try:
        results = retrieve(question, n_results=3, project_id=project_id)
    except Exception as exc:
        raise RAGUnavailableError(
            "Codebase retrieval is unavailable. Check Ollama and ChromaDB."
        ) from exc
    if not results:
        raise RAGUnavailableError(
            "The codebase index is empty. Index a repository before asking questions."
        )

    context = format_context(results)
    prompt = f"""
You are an AI software engineering assistant.

Answer the user's question using the provided codebase context.

Rules:
1. Use the provided context as your primary source.
2. Do not invent files, functions, classes, or code.
3. If the context does not contain enough information, explicitly say so.
4. Mention relevant file names and symbols, functions, or classes.
5. Explain what the retrieved code actually does.
6. Keep the answer concise but useful; do not return only a path unless that is all the context establishes.

CODEBASE CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
    answer = generate(prompt)
    sources = [
        {
            key: result["metadata"].get(key)
            for key in (
                "file", "symbol", "type", "start_line", "end_line",
                "language", "parent",
            )
        }
        for result in results
    ]
    return {"answer": answer, "sources": sources}


def answer_question(question):
    return ask_codebase(question)["answer"]


if __name__ == "__main__":

    question = input(
        "Ask about the codebase: "
    )

    result = ask_codebase(question)

    print("\nAI:\n")
    print(result["answer"])
