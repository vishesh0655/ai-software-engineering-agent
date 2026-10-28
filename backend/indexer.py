from pathlib import Path

from backend.chunking import chunk_project
from backend.embeddings import get_embeddings
from backend.vector_store import VectorStore


def create_chunk_id(chunk):
    metadata = chunk["metadata"]

    file_path = metadata["file"]
    symbol = metadata["symbol"]
    start_line = metadata["start_line"]
    end_line = metadata["end_line"]

    return (
        f"{file_path}:"
        f"{symbol}:"
        f"{start_line}-{end_line}"
    )


def index_project(project_path="."):
    print("Scanning project...")

    chunks = chunk_project(project_path)

    if not chunks:
        print("No code chunks found.")
        return

    print(f"Found {len(chunks)} chunks.")

    documents = [
        chunk["content"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    ids = [
        create_chunk_id(chunk)
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = get_embeddings(documents)

    print("Storing in ChromaDB...")

    vector_store = VectorStore()

    vector_store.add_documents(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"Indexed {len(chunks)} chunks successfully."
    )

    print(
        f"Total chunks in database: "
        f"{vector_store.count()}"
    )


if __name__ == "__main__":

    project_path = Path(".")

    index_project(project_path)
