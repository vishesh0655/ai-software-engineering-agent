import os
from pathlib import Path


CHROMA_DATA_PATH = (Path(__file__).resolve().parent.parent / "data" / "chroma").resolve()


# Folders we don't want the AI to read
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
}
IGNORED_DIRS_LOWER = {directory.lower() for directory in IGNORED_DIRS}


# File extensions we currently care about
ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".html",
    ".css",
    ".md",
    ".json",
    ".sql",
}


def get_project_files(project_path):
    project_path = Path(project_path).resolve()
    files = []

    for current_directory, directories, filenames in os.walk(project_path, topdown=True):
        current_path = Path(current_directory)
        directories[:] = [
            directory
            for directory in directories
            if directory.lower() not in IGNORED_DIRS_LOWER
            and not (directory.lower().startswith(".venv") and directory.lower() != ".venv")
            and not _is_chroma_path(current_path / directory)
        ]

        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            if _is_chroma_path(path):
                continue
            files.append(path)

    return files


def _is_chroma_path(path):
    try:
        path.resolve().relative_to(CHROMA_DATA_PATH)
        return True
    except ValueError:
        return False


if __name__ == "__main__":

    project_path = "."

    files = get_project_files(project_path)

    print(f"Found {len(files)} files:\n")

    for file in files:
        print(file)
