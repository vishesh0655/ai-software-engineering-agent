import ast
from pathlib import Path

from backend.generic_chunker import get_text_chunks
from backend.ingestion import get_project_files


def get_python_chunks(file_path):
    """
    Structure-aware chunking for Python files.

    Python functions and class methods are stored as
    individual chunks so retrieval can target specific
    pieces of code.
    """

    file_path = Path(file_path)

    try:
        source = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # If Python cannot be parsed, fall back to the
        # universal text chunker.
        return get_text_chunks(file_path)

    lines = source.splitlines()

    if not lines:
        return []

    chunks = []

    for node in tree.body:

        # ---------------------------------------------
        # Top-level functions
        # ---------------------------------------------

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):

            start_line = node.lineno

            end_line = getattr(
                node,
                "end_lineno",
                start_line,
            )

            code = "\n".join(
                lines[start_line - 1:end_line]
            )

            chunks.append({
                "content": code,
                "metadata": {
                    "file": str(file_path),
                    "language": "python",
                    "symbol": node.name,
                    "type": "function",
                    "parent": "",
                    "start_line": start_line,
                    "end_line": end_line,
                },
            })

        # ---------------------------------------------
        # Classes
        # ---------------------------------------------

        elif isinstance(node, ast.ClassDef):

            # Store individual methods instead of storing
            # the complete class again. This prevents
            # duplicated retrieval results.

            class_methods = []

            for child in node.body:

                if not isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue

                method_start = child.lineno

                method_end = getattr(
                    child,
                    "end_lineno",
                    method_start,
                )

                method_code = "\n".join(
                    lines[
                        method_start - 1:
                        method_end
                    ]
                )

                chunks.append({
                    "content": method_code,
                    "metadata": {
                        "file": str(file_path),
                        "language": "python",
                        "symbol": child.name,
                        "type": "method",
                        "parent": node.name,
                        "start_line": method_start,
                        "end_line": method_end,
                    },
                })

                class_methods.append(child)

            # If a class contains no methods, store the
            # class itself so we don't completely lose it.
            if not class_methods:

                class_start = node.lineno

                class_end = getattr(
                    node,
                    "end_lineno",
                    class_start,
                )

                class_code = "\n".join(
                    lines[
                        class_start - 1:
                        class_end
                    ]
                )

                chunks.append({
                    "content": class_code,
                    "metadata": {
                        "file": str(file_path),
                        "language": "python",
                        "symbol": node.name,
                        "type": "class",
                        "parent": "",
                        "start_line": class_start,
                        "end_line": class_end,
                    },
                })

    # ---------------------------------------------
    # Handle Python files containing only imports,
    # constants, decorators, etc.
    # ---------------------------------------------

    if not chunks:

        return get_text_chunks(file_path)

    return chunks


def chunk_file(file_path):
    """
    Universal entry point for chunking a file.

    Python gets structure-aware AST chunking.

    Every other supported file type uses the same
    universal text chunker.
    """

    file_path = Path(file_path)

    if file_path.suffix.lower() == ".py":
        return get_python_chunks(file_path)

    return get_text_chunks(file_path)


def chunk_project(project_path):
    """
    Chunk every supported file in a project.
    """

    files = get_project_files(project_path)

    all_chunks = []

    for file_path in files:

        chunks = chunk_file(file_path)

        all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":

    chunks = chunk_project(".")

    print(
        f"Total chunks: {len(chunks)}"
    )

    for i, chunk in enumerate(
        chunks[:30],
        start=1,
    ):

        metadata = chunk["metadata"]

        print("\n" + "=" * 60)

        print(f"CHUNK {i}")
        print(f"File: {metadata['file']}")
        print(
            f"Language: "
            f"{metadata['language']}"
        )
        print(
            f"Symbol: "
            f"{metadata['symbol']}"
        )
        print(
            f"Type: "
            f"{metadata['type']}"
        )
        print(
            f"Parent: "
            f"{metadata['parent']}"
        )
        print(
            f"Lines: "
            f"{metadata['start_line']}-"
            f"{metadata['end_line']}"
        )
