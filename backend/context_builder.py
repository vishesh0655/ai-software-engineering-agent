from backend.retrieval import retrieve


def format_context(results):
    context_parts = []

    for i, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        context_parts.append(
            f"""
--- SOURCE {i} ---
File: {metadata["file"]}
Symbol: {metadata["symbol"]}
Type: {metadata["type"]}
Lines: {metadata["start_line"]}-{metadata["end_line"]}

Code:
{result["content"]}
"""
        )

    return "\n".join(context_parts)


def build_context(query, n_results=3, project_id=None):
    results = retrieve(
        query,
        n_results=n_results,
        project_id=project_id,
    )
    return format_context(results)


if __name__ == "__main__":

    query = input(
        "Ask about the codebase: "
    )

    context = build_context(query)

    print("\nRetrieved Context:\n")
    print(context)
