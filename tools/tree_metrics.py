import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set


class Node:
    __slots__ = ("files", "lines", "code", "children")

    def __init__(self) -> None:
        self.files: int = 0
        self.lines: int = 0
        self.code: int = 0
        self.children: Dict[str, Node] = {}

    def add_counts(self, files: int, lines: int, code: int) -> None:
        self.files += files
        self.lines += lines
        self.code += code


# --- aggregation helpers ----------------------------------------------------


def normalize_excludes(repo: Path, raw_paths: Sequence[str]) -> Set[Path]:
    excludes: Set[Path] = set()
    for raw in raw_paths:
        candidate = Path(raw)
        rel_parts: List[str] = [
            part for part in candidate.parts if part not in ("", ".")
        ]
        if candidate.is_absolute():
            rel = candidate.relative_to(repo)
            rel_parts = [part for part in rel.parts if part not in ("", ".")]
        assert (
            rel_parts
        ), f"exclude path '{raw}' did not resolve to a valid repo-relative directory"
        top = repo / rel_parts[0]
        assert (
            top.exists() and top.is_dir()
        ), f"exclude path '{raw}' is not an existing directory"
        excludes.add(top)
    return excludes


def should_skip(path: Path, excluded_roots: Iterable[Path]) -> bool:
    for root in excluded_roots:
        if path.is_relative_to(root):
            return True
    return False


def build_tree(repo: Path, max_depth: int, excluded_roots: Iterable[Path]) -> Node:
    assert max_depth > 0, "--levels must be positive"
    root = Node()
    py_files = sorted(repo.rglob("*.py"))

    for path in py_files:
        if should_skip(path, excluded_roots):
            continue
        rel = path.relative_to(repo)
        dirs = rel.parts[:-1]
        total_lines = 0
        code_lines = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                total_lines += 1
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    code_lines += 1
        root.add_counts(1, total_lines, code_lines)

        node = root
        for depth, part in enumerate(dirs, start=1):
            if depth > max_depth:
                break
            child = node.children.get(part)
            if child is None:
                child = Node()
                node.children[part] = child
            node = child
            node.add_counts(1, total_lines, code_lines)
    return root


def emit_tree(name: str, node: Node, depth: int, max_depth: int) -> None:
    indent = "  " * depth
    print(f"{indent}- {name}: files={node.files}, lines={node.lines}, code={node.code}")
    if depth >= max_depth:
        return
    for child_name in sorted(node.children):
        emit_tree(child_name, node.children[child_name], depth + 1, max_depth)


# --- cli --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Python file counts by directory tree."
    )
    parser.add_argument(
        "--levels", type=int, default=4, help="maximum directory depth to display"
    )
    parser.add_argument(
        "--excludes",
        nargs="*",
        default=[],
        help="top-level repo directories to exclude (e.g. ./configs configs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path.cwd()
    excluded = normalize_excludes(repo, args.excludes)
    tree = build_tree(repo, args.levels, excluded)
    emit_tree(".", tree, depth=0, max_depth=args.levels)


if __name__ == "__main__":
    main()
