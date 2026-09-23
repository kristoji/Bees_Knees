"""Check that two revisions of src/ produce the same engine fingerprint.

    python bench/diffcheck.py <rev-A> [rev-B]      # rev-B defaults to the working tree

Exits non-zero when the digests differ, so it can gate a refactor of move generation.
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WORKTREE = ("WORK", "worktree", ".")


def export_src(rev, dest):
    tree = os.path.join(dest, "tree")
    if rev in WORKTREE:
        shutil.copytree(os.path.join(ROOT, "src"), tree)
        return tree
    tar = subprocess.run(["git", "archive", rev, "src"], cwd=ROOT,
                         check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", dest], input=tar, check=True)
    os.rename(os.path.join(dest, "src"), tree)
    return tree


def fingerprint_of(rev, tmp):
    src = export_src(rev, tmp)
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "bench", "difftest.py")],
        env=dict(os.environ, BENCH_SRC=src), capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit(f"difftest failed on {rev}")
    return out.stdout.strip()


def main():
    rev_a = sys.argv[1]
    rev_b = sys.argv[2] if len(sys.argv) > 2 else "."
    tmp_a, tmp_b = tempfile.mkdtemp(), tempfile.mkdtemp()
    try:
        a = fingerprint_of(rev_a, tmp_a)
        b = fingerprint_of(rev_b, tmp_b)
    finally:
        shutil.rmtree(tmp_a, ignore_errors=True)
        shutil.rmtree(tmp_b, ignore_errors=True)
    print(f"{rev_a:>12}  {a}")
    print(f"{rev_b:>12}  {b}")
    if a == b:
        print("MATCH")
    else:
        print("DIFFER")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
