"""Score a checkpoint's policy head: top-1/3/5 and the rank of the move actually played.

Runs against a checkpoint on disk, so it can be pointed at the best.pt of a training
job that is still going without disturbing it.

Top-1 alone understates what the search gets out of a prior: MCTS does not need the
first guess to be right, it needs the probability concentrated on a few sensible
candidates instead of spread flat over sixty. The rank distribution says that directly.

    python tools/eval_policy.py --shards data/shards --weights models/agent/best.pt \
        --summary models/agent/summary.json --split val
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from ai.agent_net import AgentNet, segment_log_softmax  # noqa: E402
from train_agent import Corpus, bce, decisive_accuracy, split_by_game  # noqa: E402


@torch.no_grad()
def policy_ranks(model, corpus, idx, batch_size, amp_dtype):
    """Rank of the played move among the legal ones, 0 = the head's first choice."""
    ranks, values = [], []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        (x, edge_index, batch_vec, _, src, bug, dst,
         move_seg, target, has_target) = corpus.batch(chunk)
        with torch.autocast(corpus.device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            value, move_logits = model(x, edge_index, batch_vec, src, bug, dst, move_seg)
        values.append(torch.sigmoid(value.float()))
        if not has_target.any():
            continue
        flat = move_logits.float()
        chosen = target[has_target]
        # How many legal moves the head scores strictly above the played one.
        # flat[chosen] has one entry per graph WITH a target, while move_seg indexes
        # every graph in the chunk, so scatter it into a full-length vector first.
        # +inf for the targetless graphs makes their moves count as "not better".
        per_graph = torch.full((len(chunk),), float("inf"), device=x.device,
                               dtype=torch.float32)
        per_graph[has_target] = flat[chosen]
        better = (flat > per_graph[move_seg]).to(torch.float32)
        count = torch.zeros(len(chunk), device=x.device, dtype=torch.float32)
        count = count.index_add(0, move_seg, better)
        ranks.append(count[has_target].to(torch.int64))
    return (torch.cat(ranks) if ranks else torch.zeros(0, dtype=torch.int64),
            torch.cat(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = json.load(open(args.summary))["args"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = None
    if device.type == "cuda":
        amp = (torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8
               else torch.float16)

    corpus = Corpus(args.shards, cfg.get("collections"), device)
    splits = split_by_game(corpus.game_key, cfg["seed"])
    idx = splits[args.split].to(device)

    model = AgentNet(
        in_dim=corpus.num_features, hidden_dim=cfg["hidden_dim"],
        conv_type=cfg["conv_type"], num_layers=cfg["num_layers"],
        gat_heads=cfg["gat_heads"], conv_dropout=cfg["dropout"],
        mlp_dropout=cfg["dropout"], final_dropout=cfg["dropout"],
        use_layer_norm=True, use_residual=False, pooling=cfg["pooling"],
        mlp_layers=2, final_mlp_layers=2,
    ).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    ranks, values = policy_ranks(model, corpus, idx, args.batch_size, amp)
    target = corpus.y.index_select(0, idx)
    n = len(ranks)

    n_legal = (corpus.move_ptr.index_select(0, idx + 1)
               - corpus.move_ptr.index_select(0, idx)).float()

    print(f"split {args.split}: {len(idx)} positions, {n} with a played move")
    print(f"legal moves per position: mean {n_legal.mean():.1f}, max {int(n_legal.max())}")
    print()
    print(f"{'k':>3}  {'top-k':>7}  {'random':>7}  {'ratio':>6}")
    for k in (1, 3, 5, 10, 20):
        hit = float((ranks < k).float().mean())
        chance = float((torch.clamp(n_legal[:n] if n else n_legal, min=1)
                        .reciprocal() * k).clamp(max=1.0).mean())
        print(f"{k:>3}  {hit * 100:>6.1f}%  {chance * 100:>6.1f}%  {hit / max(1e-9, chance):>5.1f}x")
    print()
    print(f"median rank of the played move: {int(ranks.float().median())}")
    print(f"value: BCE {bce(values, target):.4f}  acc {decisive_accuracy(values, target):.3f}")


if __name__ == "__main__":
    main()
