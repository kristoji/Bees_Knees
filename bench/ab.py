"""A/B two revisions of src/ through the same benchmark.

Runs alternate so CPU frequency scaling hits both equally, and the best of N
rounds is reported: on this machine a single round is bimodal by ~2.7x, so a
one-shot comparison can invent or hide a speedup entirely.

    python bench/ab.py <rev-A> [rev-B]      # rev-B defaults to HEAD
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
METRICS = [
    ("Board() construction", "ms", "min"),
    ("safe_play+undo  pos1", "us", "min"),
    ("get_valid_moves (cold)  pos1", "us", "min"),
    ("MCTS rollouts/s  TOTAL", "/s", "max"),
]


WORKTREE = ("WORK", "worktree", ".")


def export_src(rev, dest):
    """Materialise src/ at `rev`. The literal "." means the working tree."""
    tree = os.path.join(dest, "tree")
    if rev in WORKTREE:
        shutil.copytree(os.path.join(ROOT, "src"), tree)
        return tree
    tar = subprocess.run(["git", "archive", rev, "src"], cwd=ROOT,
                         check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=tar, check=True)
    os.rename(os.path.join(dest, "src"), tree)
    return tree


def run(src, rollouts):
    env = dict(os.environ, BENCH_SRC=src)
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "bench", "bench_engine.py"),
         "--repeat", "1", "--rollouts", str(rollouts)],
        env=env, check=True, capture_output=True, text=True).stdout
    values = {}
    for line in out.splitlines():
        for label, _unit, _how in METRICS:
            if line.startswith(label):
                values[label] = float(re.findall(r"[-+0-9.]+", line[len(label):])[0])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rev_a")
    parser.add_argument("rev_b", nargs="?", default=".",
                        help='"." (the default) means the working tree')
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=500)
    args = parser.parse_args()

    tmp_a = tempfile.mkdtemp()
    tmp_b = tempfile.mkdtemp()
    try:
        src_a = export_src(args.rev_a, tmp_a)
        src_b = export_src(args.rev_b, tmp_b)
        results = {args.rev_a: [], args.rev_b: []}
        for i in range(args.rounds):
            results[args.rev_a].append(run(src_a, args.rollouts))
            results[args.rev_b].append(run(src_b, args.rollouts))
            print(f"round {i + 1}/{args.rounds} done", file=sys.stderr)

        print(f"\n{'metric':<32}{args.rev_a:>14}{args.rev_b:>14}   ratio")
        print("-" * 76)
        for label, unit, how in METRICS:
            pick = min if how == "min" else max
            a = pick(r[label] for r in results[args.rev_a])
            b = pick(r[label] for r in results[args.rev_b])
            ratio = (a / b) if how == "min" else (b / a)
            print(f"{label + ' (' + unit + ')':<32}{a:>14.3f}{b:>14.3f}{ratio:>8.2f}x")
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)


if __name__ == "__main__":
    main()
