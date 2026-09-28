"""Sum the per-shard results of a duel job array into one verdict.

    python tools/aggregate_duels.py /path/to/duel-*.json

A raw score means little without its uncertainty: at 100 games the standard error is
about 5 points, so a 55% result and a dead heat are the same measurement. The Wilson
interval is printed for that reason, and the Elo difference only as a reading of the
score, not as an independent fact.
"""
import glob
import json
import math
import os
import sys


def wilson(wins, draws, games, z=1.96):
    """Confidence interval for the score, counting a draw as half a win."""
    if games == 0:
        return 0.0, 0.0
    p = (wins + 0.5 * draws) / games
    d = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / d
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def elo(score):
    """Elo difference implied by a score, which is undefined at a clean sweep."""
    if score <= 0 or score >= 1:
        return None
    return -400 * math.log10(1 / score - 1)


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    # Accept a directory as well as a glob, because a shell that matched nothing and a
    # directory both used to end in an unhelpful traceback.
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            paths.extend(sorted(glob.glob(os.path.join(arg, "*.json"))))
        else:
            paths.append(arg)
    if not paths:
        raise SystemExit(f"no result files in {args}")

    total = {"wins": 0, "losses": 0, "draws": 0, "capped": 0, "games": 0, "seconds": 0.0}
    rollouts, opponents = set(), set()
    for path in paths:
        d = json.load(open(path))
        for key in ("wins", "losses", "draws", "capped", "games", "seconds"):
            total[key] += d[key]
        rollouts.add(d["rollouts"])
        opponents.add(os.path.basename(os.path.dirname(d.get("opponent", "heuristic")))
                      or d.get("opponent", "heuristic"))

    games = total["games"]
    decided = total["wins"] + total["losses"]
    drawn = total["draws"] + total["capped"]
    # Reaching the ply cap counts as a draw, by the duel's convention.
    score = (total["wins"] + 0.5 * drawn) / max(1, games)
    name = "/".join(sorted(opponents)) if opponents else "opponent"
    print(f"{len(paths)} shards, {games} games at {sorted(rollouts)} rollouts per move")
    print(f"  A wins         {total['wins']}")
    print(f"  {name[:14]:14s} {total['losses']}")
    print(f"  draws          {drawn}"
          f"   ({total['draws']} on the board, {total['capped']} at the ply cap)")
    lo, hi = wilson(total["wins"], drawn, games)
    print(f"  score          {score * 100:.1f}%   95% CI [{lo * 100:.1f}, {hi * 100:.1f}]")
    e, elo_lo, elo_hi = elo(score), elo(lo), elo(hi)
    if e is not None:
        span = ("?" if elo_lo is None else f"{elo_lo:+.0f}",
                "?" if elo_hi is None else f"{elo_hi:+.0f}")
        print(f"  Elo            {e:+.0f}   95% CI [{span[0]}, {span[1]}]")
    if lo <= 0.5 <= hi:
        print("  the interval contains 50%: this does not separate the two players")
    if decided:
        print(f"  among decided  {total['wins']}/{decided} = "
              f"{total['wins'] / decided * 100:.1f}%")
    else:
        print("  no game was decided: the ply cap or draws absorbed them all")
    print(f"  compute        {total['seconds'] / 3600:.1f} core-hours")


if __name__ == "__main__":
    main()
