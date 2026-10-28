from backend.embeddings import get_embeddings
from backend.vector_store import VectorStore


def is_redundant(new_result, selected_results):
    new_metadata = new_result["metadata"]

    for selected in selected_results:
        selected_metadata = selected["metadata"]

        # Same file + overlapping lines = redundant
        if (
            new_metadata["file"] == selected_metadata["file"]
            and not (
                new_metadata["end_line"] < selected_metadata["start_line"]
                or new_metadata["start_line"] > selected_metadata["end_line"]
            )
        ):
            return True

    return False


def retrieve(query, n_results=3, project_id=None):
    if n_results <= 0:
        return []

    query_embedding = get_embeddings([query])[0]

    vector_store = VectorStore(project_id=project_id)

    # Retrieve extra candidates because
    # some will be removed as redundant.
    candidate_count = min(n_results * 2, vector_store.count())
    if candidate_count == 0:
        return []

    results = vector_store.search(
        query_embedding=query_embedding,
        n_results=candidate_count,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    selected_results = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        result = {
            "content": document,
            "metadata": metadata,
            "distance": distance,
        }

        if is_redundant(result, selected_results):
            continue

        selected_results.append(result)

        if len(selected_results) >= n_results:
            break

    return selected_results


if __name__ == "__main__":

    query = input(
        "Ask about the codebase: "
    )

    results = retrieve(query)

    print("\nRelevant code:\n")

    for i, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print("=" * 60)

        print(f"Result {i}")
        print(f"File: {metadata['file']}")
        print(f"Symbol: {metadata['symbol']}")
        print(f"Type: {metadata['type']}")
        print(f"Parent: {metadata['parent']}")

        print(
            f"Lines: "
            f"{metadata['start_line']}-"
            f"{metadata['end_line']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print("\nCode:")
        print(result["content"])
