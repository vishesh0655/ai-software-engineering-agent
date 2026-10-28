from backend.ingestion import get_project_files


def test_project_scan_skips_versioned_python_environments(tmp_path):
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("def main():\n    return True\n", encoding="utf-8")
    environment_file = tmp_path / ".venv313" / "Lib" / "site-packages" / "package.py"
    environment_file.parent.mkdir(parents=True)
    environment_file.write_text("def dependency():\n    pass\n", encoding="utf-8")

    files = get_project_files(tmp_path)

    assert files == [source]
