"""Sum the per-shard results of a duel job array into one verdict.

    python tools/aggregate_duels.py /path/to/duel-*.json
"""
import json
import sys


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    total = {"wins": 0, "losses": 0, "draws": 0, "capped": 0, "games": 0, "seconds": 0.0}
    rollouts = set()
    for path in paths:
        d = json.load(open(path))
        for key in ("wins", "losses", "draws", "capped", "games", "seconds"):
            total[key] += d[key]
        rollouts.add(d["rollouts"])

    games = total["games"]
    decided = total["wins"] + total["losses"]
    # Reaching the ply cap counts as a draw, by the duel's convention.
    score = (total["wins"] + 0.5 * (total["draws"] + total["capped"])) / max(1, games)
    print(f"{len(paths)} shards, {games} games at {sorted(rollouts)} rollouts per move")
    print(f"  GNN wins       {total['wins']}")
    print(f"  heuristic wins {total['losses']}")
    print(f"  draws          {total['draws'] + total['capped']}"
          f"   ({total['draws']} on the board, {total['capped']} at the ply cap)")
    print(f"  score          {score * 100:.1f}%")
    if decided:
        print(f"  among decided  {total['wins']}/{decided} = "
              f"{total['wins'] / decided * 100:.1f}%")
    else:
        print("  no game was decided: the ply cap or draws absorbed them all")
    print(f"  compute        {total['seconds'] / 3600:.1f} core-hours")


if __name__ == "__main__":
    main()
