import warnings
from pathlib import Path


def test_backend_sources_compile_without_invalid_escape_syntax_warnings():
    backend_root = Path(__file__).resolve().parents[1] / "backend"
    reported = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        for source_path in backend_root.rglob("*.py"):
            source = source_path.read_text(encoding="utf-8")
            compile(source, str(source_path), "exec")
        reported = [str(item.message) for item in caught if issubclass(item.category, SyntaxWarning)]

    assert reported == []
