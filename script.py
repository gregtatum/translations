import subprocess
from pathlib import Path

DOCS_DIR = Path("tracking")
OUTPUT_FILE = Path("tracking.txt")
INCLUDE_EXTENSIONS = {".py", ".sh", ".in", "*.md", "*.yml", "*.perl"}


def build_file_list(base_dir):
    return sorted(
        [
            p
            for p in base_dir.rglob("*")
            if p.suffix in INCLUDE_EXTENSIONS and p.is_file()
        ]
    )


def build_file_tree_with_unix_tree(base_dir):
    result = subprocess.run(
        ["TREE", "."],
        cwd=base_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return f"{base_dir}\n" + result.stdout.strip()


def concat_files_with_headers(file_paths, base_dir):
    chunks = []
    for path in file_paths:
        rel_path = path.relative_to(base_dir)
        chunks.append(str(rel_path))
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(chunks)


def main():
    files = build_file_list(DOCS_DIR)
    tree = build_file_tree_with_unix_tree(DOCS_DIR)
    body = concat_files_with_headers(files, DOCS_DIR)
    OUTPUT_FILE.write_text(tree + "\n\n" + body, encoding="utf-8")


if __name__ == "__main__":
    main()
