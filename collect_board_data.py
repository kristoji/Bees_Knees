# Script to collect all txt files named board.txt inside a directory in a path to specify via command line(they might be nested inside subfolders) inside a dataset folder in a path to specify via command line. rename all this files to board_folder_game<index>.txt

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


def find_board_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("board.txt") if p.is_file())


def find_matching_ancestor(path: Path, pattern: str, mode: str) -> Optional[Path]:
    """Return the nearest ancestor (including the immediate parent) whose name matches pattern.

    mode: one of 'contains', 'startswith', 'endswith', 'exact', 'regex'
    """
    if not pattern:
        return None

    cur = path.parent
    while True:
        if cur == cur.parent:
            # reached filesystem root
            return None

        name = cur.name
        matched = False
        if mode == "contains":
            matched = pattern in name
        elif mode == "startswith":
            matched = name.startswith(pattern)
        elif mode == "endswith":
            matched = name.endswith(pattern)
        elif mode == "exact":
            matched = name == pattern
        elif mode == "regex":
            try:
                matched = re.search(pattern, name) is not None
            except re.error:
                matched = False

        if matched:
            return cur

        if cur.parent == cur:
            return None
        cur = cur.parent


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def next_available_dest(dataset_dir: Path, parent_name: str, start_index: int, pad: int) -> Path:
    idx = start_index
    while True:
        candidate = dataset_dir / f"board_{parent_name}_game{idx:0{pad}d}.txt"
        if not candidate.exists():
            return candidate
        idx += 1


def collect(source_dir: Path, dataset_dir: Path, move: bool, start_index: int, pad: int, dry_run: bool,
        ancestor_pattern: Optional[str] = None, ancestor_mode: str = "contains") -> int:
    files = find_board_files(source_dir)
    if not files:
        print("No board.txt files found.")
        return 0

    ensure_dir(dataset_dir)

    count = 0
    current_index = start_index
    for src in files:
        # If an ancestor pattern was provided, try to find the nearest matching ancestor and
        # use its name both for filtering and for the destination filename. If no matching
        # ancestor is found, skip this file.
        if ancestor_pattern:
            match = find_matching_ancestor(src, ancestor_pattern, ancestor_mode)
            if match is None:
                # skip files not under a matching ancestor
                print(f"SKIP (no matching ancestor): {src}")
                continue
            parent_name = match.name
        else:
            parent_name = src.parent.name or "root"
        dest = next_available_dest(dataset_dir, parent_name, current_index, pad)

        action = "MOVE" if move else "COPY"
        print(f"{action}: {src} -> {dest}")
        if not dry_run:
            if move:
                shutil.move(str(src), str(dest))
            else:
                shutil.copy2(str(src), str(dest))

        count += 1
        current_index += 1

    print(f"Done. {count} file(s) processed.")
    return count


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect all 'board.txt' files recursively from SOURCE_DIR and copy/move them to DATASET_DIR "
                    "renaming to board_<parentFolder>_game<index>.txt"
    )
    parser.add_argument("source_dir", type=Path, help="Root directory to search")
    parser.add_argument("dataset_dir", type=Path, help="Destination dataset directory")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--start-index", type=int, default=1, help="Starting index (default: 1)")
    parser.add_argument("--pad", type=int, default=3, help="Zero-padding width for index (default: 3 -> 001, 002, ...)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    parser.add_argument("--ancestor-pattern", type=str, default=None,
                        help="If set, only collect files whose ancestor directory matches this pattern (e.g. 'tournament').")
    parser.add_argument("--ancestor-mode", type=str, default="contains",
                        choices=["contains", "startswith", "endswith", "exact", "regex"],
                        help="How to match the ancestor pattern. Default: contains")

    args = parser.parse_args(argv)

    source = args.source_dir.resolve()
    dataset = args.dataset_dir.resolve()

    if not source.exists() or not source.is_dir():
        raise SystemExit(f"Source directory does not exist or is not a directory: {source}")

    collect(source, dataset, move=args.move, start_index=args.start_index, pad=args.pad,
            dry_run=args.dry_run, ancestor_pattern=args.ancestor_pattern, ancestor_mode=args.ancestor_mode)


if __name__ == "__main__":
    main()

#python collect_board_data.py "C:\Users\Alberto\Downloads\data" "C:\Users\Alberto\Desktop\NN\Bees_Knees\pro_matches\board_data_tournaments" --ancestor-pattern '_tournament$' --ancestor-mode regex
