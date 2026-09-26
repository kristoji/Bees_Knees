"""The train/val/test split, in one place because it must not drift between callers.

The split assigns each GAME to one bucket, so no position of a game is ever in two
splits, and it derives the bucket from the game's own key rather than from its position
in a shuffle. That distinction matters: the earlier version permuted game indices, so
its permutation depended on how many games there were, and adding a collection
reshuffled every split. A model trained on the larger corpus had then seen most of the
smaller corpus's test games, which makes two such models incomparable. With a per-game
hash, a game's bucket is a property of the game and the seed alone, so a corpus can
grow and every existing game stays where it was.

The cost is that the splits are no longer exactly 10%: each game lands in a bucket
independently, so the sizes vary around the requested fractions the way a binomial
does. Reported split sizes are therefore approximate by design.
"""
import numpy as np
import torch

# splitmix64's multipliers and finaliser. Consecutive game ids differ in their low
# bits only, and this is what spreads them across the whole range.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def game_fraction(games, seed):
    """A value in [0, 1) per game, depending on the game key and the seed alone."""
    with np.errstate(over="ignore"):
        # numpy uint64 wraps silently and shifts logically, which is what splitmix64
        # is defined over; torch int64 would shift arithmetically and sign-extend.
        h = np.asarray(games.cpu().numpy(), dtype=np.uint64)
        h = h * _GOLDEN + np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
        h = h ^ (h >> np.uint64(30))
        h = h * _MIX1
        h = h ^ (h >> np.uint64(27))
        h = h * _MIX2
        h = h ^ (h >> np.uint64(31))
    # Keep the top 53 bits, the most mixed ones, and the most a float64 holds exactly.
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def split_by_game(game_key, seed, val_frac=0.1, test_frac=0.1):
    """Partition positions so that no game appears in more than one split."""
    games = torch.unique(game_key)
    frac = torch.from_numpy(game_fraction(games, seed)).to(games.device)
    bounds = {"test": (0.0, test_frac),
              "val": (test_frac, test_frac + val_frac),
              "train": (test_frac + val_frac, 1.0)}
    out = {}
    for name, (lo, hi) in bounds.items():
        chosen = games[(frac >= lo) & (frac < hi)]
        out[name] = torch.nonzero(torch.isin(game_key, chosen), as_tuple=True)[0]
    return out
