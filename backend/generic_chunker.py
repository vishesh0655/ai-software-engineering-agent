from pathlib import Path


# Maximum number of lines in one generic chunk.
DEFAULT_CHUNK_SIZE = 80

# Number of lines carried from the previous chunk.
# This helps preserve context when a function/class is split
# across two chunks.
DEFAULT_OVERLAP = 10


def get_text_chunks(
    file_path,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_OVERLAP,
):
    """
    Generic chunker for text/code files.

    This works for JavaScript, TypeScript, Java, C, C++,
    HTML, CSS, JSON, SQL, Markdown, and other text files.

    It does not try to understand the language syntax.
    Instead, it splits the file into overlapping line-based
    chunks while preserving useful metadata.
    """

    file_path = Path(file_path)

    try:
        source = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return []

    if not source.strip():
        return []

    lines = source.splitlines()

    if not lines:
        return []

    # Prevent invalid configurations.
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    chunks = []

    start = 0

    while start < len(lines):

        end = min(
            start + chunk_size,
            len(lines),
        )

        code = "\n".join(
            lines[start:end]
        )

        if code.strip():

            extension = file_path.suffix.lower()

            if extension == ".py":
                language = "python"
            elif extension:
                language = extension.lstrip(".")
            else:
                language = "text"

            chunks.append({
                "content": code,
                "metadata": {
                    "file": str(file_path),
                    "language": language,
                    "symbol": "",
                    "type": "text_chunk",
                    "parent": "",
                    "start_line": start + 1,
                    "end_line": end,
                },
            })

        # Last chunk reached.
        if end >= len(lines):
            break

        # Move forward while preserving overlap.
        start = end - overlap

    return chunks


if __name__ == "__main__":

    file_path = input(
        "Enter file path: "
    )

    chunks = get_text_chunks(file_path)

    print(
        f"\nFound {len(chunks)} chunks.\n"
    )

    for i, chunk in enumerate(
        chunks,
        start=1,
    ):

        metadata = chunk["metadata"]

        print("=" * 60)

        print(f"CHUNK {i}")
        print(f"File: {metadata['file']}")
        print(f"Language: {metadata['language']}")
        print(f"Type: {metadata['type']}")

        print(
            f"Lines: "
            f"{metadata['start_line']}-"
            f"{metadata['end_line']}"
        )

        print("\nContent:")
        print(chunk["content"])