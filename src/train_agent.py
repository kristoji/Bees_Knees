"""Train value and policy together on the rebuilt shards.

Same data and same split-by-game discipline as train_value.py, plus the policy head
from ai.agent_net. The policy target is the move the pro/engine game actually played,
as a cross entropy over that position's legal moves.

Why it matters more than a better value: without a policy the search gets its prior by
evaluating the value net on every child, b+1 passes per expansion with b averaging 62.6
here. The policy head makes that one pass.

Deliberately does not go through OracleGNN: that constructor calls torch.compile with
mode="reduce-overhead", whose CUDA graphs fight with the varying batch shapes of graph
data, and its training path moves the whole dataset onto the GPU one small tensor at a
time. Here the shards are already collated, so the whole corpus is two large tensors that
can sit on the device, and a batch is a gather.

Two things it does that the old path did not:

  - splits by GAME, not by position. Every position of a game shares one outcome, so a
    per-position split puts near-identical, identically-labelled examples on both sides
    and makes the validation loss meaningless. See TRAINING_and_DATASET.md section 4.1.
  - reports the baselines. The corpus carries the hand-written heuristic's value for
    every position, so the log says whether the network actually beats counting queen
    neighbours. If it does not, it has learned nothing useful for the search.

    python src/train_value.py --shards data/shards --out models/value --hidden-dim 64
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai.agent_net import AgentNet, segment_log_softmax  # noqa: E402


# ---------------------------------------------------------------- data

class Corpus:
    """All shards concatenated, pre-collated, optionally resident on the device."""

    def __init__(self, shard_dir, collections=None, device="cpu", limit_games=0):
        xs, edges, ys, heur, game_keys, node_counts, edge_counts = [], [], [], [], [], [], []
        m_src, m_bug, m_dst, m_played, move_counts = [], [], [], [], []
        m_pi = []
        collection_ids = {}
        for path in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
            collection = os.path.basename(path)[:-4].rsplit(".", 1)[0]
            if collections and collection not in collections:
                continue
            cid = collection_ids.setdefault(collection, len(collection_ids))
            data = np.load(path)
            xs.append(data["x"])
            edges.append(data["edge_index"])
            ys.append(data["y"])
            heur.append(data["heuristic"])
            node_counts.append(np.diff(data["node_ptr"]))
            edge_counts.append(np.diff(data["edge_ptr"]))
            move_counts.append(np.diff(data["move_ptr"]))
            m_src.append(data["move_src"])
            m_bug.append(data["move_bug"])
            m_dst.append(data["move_dst"])
            m_played.append(data["move_played"])
            n_moves = len(data["move_src"])
            if "move_pi" in data.files:
                m_pi.append(data["move_pi"].astype(np.float32))
            else:
                # Supervised shards name one played move; turn it into the same
                # per-move distribution the self-play shards carry, so the loss has a
                # single form.
                pi = np.zeros(n_moves, dtype=np.float32)
                ptr, played = data["move_ptr"], data["move_played"]
                hit = played >= 0
                pi[ptr[:-1][hit] + played[hit]] = 1.0
                m_pi.append(pi)
            # Game ids restart per collection, so key on the pair.
            game_keys.append(data["game_id"].astype(np.int64) + cid * 10_000_000)
        if not xs:
            raise SystemExit(f"no shards in {shard_dir} matching {collections}")

        self.collections = sorted(collection_ids, key=collection_ids.get)
        x = np.concatenate(xs)
        edge_index = np.concatenate(edges, axis=1)
        y = np.concatenate(ys)
        self.heuristic = torch.from_numpy(np.concatenate(heur))
        game_key = np.concatenate(game_keys)

        node_ptr = np.zeros(len(y) + 1, dtype=np.int64)
        edge_ptr = np.zeros(len(y) + 1, dtype=np.int64)
        move_ptr = np.zeros(len(y) + 1, dtype=np.int64)
        np.cumsum(np.concatenate(node_counts), out=node_ptr[1:])
        np.cumsum(np.concatenate(edge_counts), out=edge_ptr[1:])
        np.cumsum(np.concatenate(move_counts), out=move_ptr[1:])

        if limit_games:
            raise SystemExit("--limit-games is not supported here; use --limit-fraction")
        if False:
            keep_games = np.unique(game_key)[:limit_games]
            mask = np.isin(game_key, keep_games)
            keep = np.flatnonzero(mask)
            x, edge_index, node_ptr, edge_ptr = _subset(x, edge_index, node_ptr, edge_ptr, keep)
            y, game_key = y[keep], game_key[keep]
            self.heuristic = self.heuristic[torch.from_numpy(keep)]

        self.device = torch.device(device)
        self.x = torch.from_numpy(x).to(self.device)
        self.edge_index = torch.from_numpy(edge_index).to(self.device)
        self.node_ptr = torch.from_numpy(node_ptr).to(self.device)
        self.edge_ptr = torch.from_numpy(edge_ptr).to(self.device)
        self.y = torch.from_numpy(y).to(self.device)
        self.game_key = torch.from_numpy(game_key)
        self.num_features = x.shape[1]
        self.move_ptr = torch.from_numpy(move_ptr).to(self.device)
        self.move_src = torch.from_numpy(np.concatenate(m_src).astype(np.int64)).to(self.device)
        self.move_bug = torch.from_numpy(np.concatenate(m_bug).astype(np.int64)).to(self.device)
        self.move_dst = torch.from_numpy(np.concatenate(m_dst).astype(np.int64)).to(self.device)
        self.move_played = torch.from_numpy(
            np.concatenate(m_played).astype(np.int64)).to(self.device)
        self.move_pi = torch.from_numpy(np.concatenate(m_pi)).to(self.device)

    def __len__(self):
        return len(self.y)

    def bytes(self):
        return (self.x.numel() * 4 + self.edge_index.numel() * 8
                + self.node_ptr.numel() * 8 + self.edge_ptr.numel() * 8)

    def batch(self, idx):
        """Gather a batch into (x, edge_index, batch_vector, y, move tensors)."""
        node_index, node_seg, new_node_ptr = _gather(self.node_ptr, idx)
        edge_pos, edge_seg, _ = _gather(self.edge_ptr, idx)
        x = self.x.index_select(0, node_index)
        # Stored edges are local to their own graph, so they only need the new offset.
        edge_index = self.edge_index.index_select(1, edge_pos) + new_node_ptr[:-1][edge_seg]

        move_pos, move_seg, new_move_ptr = _gather(self.move_ptr, idx)
        offset = new_node_ptr[:-1][move_seg]
        src = self.move_src.index_select(0, move_pos)
        dst = self.move_dst.index_select(0, move_pos)
        # -1 means "in hand" for src and padding for dst, and must survive the offset.
        src = torch.where(src >= 0, src + offset, src)
        dst = torch.where(dst >= 0, dst + offset.unsqueeze(-1), dst)
        bug = self.move_bug.index_select(0, move_pos)

        target_pi = self.move_pi.index_select(0, move_pos)
        # A position has a usable policy target when its distribution carries any mass.
        mass = torch.zeros(len(idx), device=idx.device, dtype=target_pi.dtype)
        mass = mass.index_add(0, move_seg, target_pi)
        has_target = mass > 0
        return (x, edge_index, node_seg, self.y.index_select(0, idx),
                src, bug, dst, move_seg, target_pi, has_target)


def _gather(ptr, idx):
    """Index and segment vectors that read the variable-length slices named by idx."""
    starts = ptr.index_select(0, idx)
    counts = ptr.index_select(0, idx + 1) - starts
    out_ptr = torch.zeros(len(idx) + 1, dtype=torch.long, device=ptr.device)
    torch.cumsum(counts, 0, out=out_ptr[1:])
    seg = torch.repeat_interleave(
        torch.arange(len(idx), device=ptr.device), counts)
    within = torch.arange(int(out_ptr[-1]), device=ptr.device) - out_ptr[seg]
    return starts[seg] + within, seg, out_ptr


def _subset(x, edge_index, node_ptr, edge_ptr, keep):
    xs = [x[node_ptr[i]:node_ptr[i + 1]] for i in keep]
    es = [edge_index[:, edge_ptr[i]:edge_ptr[i + 1]] for i in keep]
    new_node = np.zeros(len(keep) + 1, dtype=np.int64)
    new_edge = np.zeros(len(keep) + 1, dtype=np.int64)
    np.cumsum([a.shape[0] for a in xs], out=new_node[1:])
    np.cumsum([e.shape[1] for e in es], out=new_edge[1:])
    return np.concatenate(xs), np.concatenate(es, axis=1), new_node, new_edge


def split_by_game(game_key, seed, val_frac=0.1, test_frac=0.1):
    """Partition positions so that no game appears in more than one split."""
    games = torch.unique(game_key)
    order = torch.randperm(len(games), generator=torch.Generator().manual_seed(seed))
    games = games[order]
    n_val = max(1, int(len(games) * val_frac))
    n_test = max(1, int(len(games) * test_frac))
    groups = {"test": games[:n_test],
              "val": games[n_test:n_test + n_val],
              "train": games[n_test + n_val:]}
    return {name: torch.nonzero(torch.isin(game_key, g), as_tuple=True)[0]
            for name, g in groups.items()}


# ---------------------------------------------------------------- metrics

def bce(pred, target):
    """Binary cross entropy against a soft target, from probabilities."""
    p = pred.clamp(1e-6, 1 - 1e-6)
    return -(target * p.log() + (1 - target) * (1 - p).log()).mean().item()


def decisive_accuracy(pred, target):
    """Accuracy over the non-draw positions only; a draw has no right answer at 0.5."""
    mask = target != 0.5
    if not mask.any():
        return float("nan")
    return ((pred[mask] > 0.5) == (target[mask] > 0.5)).float().mean().item()


@torch.no_grad()
def evaluate(model, corpus, idx, batch_size, amp_dtype):
    model.eval()
    preds, pol_loss, pol_hits, pol_n, uniform = [], 0.0, 0, 0, 0.0
    pol_hits3 = pol_hits5 = 0
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        (x, edge_index, batch_vec, _, src, bug, dst,
         move_seg, target_pi, has_target) = corpus.batch(chunk)
        with torch.autocast(corpus.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            value, move_logits = model(x, edge_index, batch_vec, src, bug, dst, move_seg)
        preds.append(torch.sigmoid(value.float()))
        if has_target.any():
            log_p = segment_log_softmax(move_logits.float(), move_seg, len(chunk))
            nll = torch.zeros(len(chunk), device=x.device, dtype=log_p.dtype)
            nll = nll.index_add(0, move_seg, -target_pi * log_p)
            pol_loss += float(nll[has_target].sum())
            # "The" target move is the one the distribution favours most.
            best_pi = torch.full((len(chunk),), -1.0, device=x.device, dtype=target_pi.dtype)
            best_pi = best_pi.scatter_reduce(0, move_seg, target_pi, reduce="amax",
                                             include_self=False)
            is_best = target_pi >= best_pi[move_seg] - 1e-9
            chosen = torch.zeros(len(chunk), dtype=torch.long, device=x.device)
            chosen = chosen.scatter(0, move_seg[is_best],
                                    torch.arange(len(target_pi), device=x.device)[is_best])
            # Rank of the target move: how many legal moves the head scores strictly
            # above it. Top-1 alone understates the prior's value to the search, which
            # only needs the right move among the first few candidates.
            # float32 explicitly: under autocast move_logits is float16, and
            # scatter_reduce requires self and src to share a dtype.
            flat = move_logits.float()
            per_graph = torch.full((len(chunk),), float("inf"), device=x.device,
                                   dtype=torch.float32)
            # chosen holds one move index per graph, so select the ones that
            # actually have a target before writing them in.
            per_graph[has_target] = flat[chosen[has_target]]
            better = (flat > per_graph[move_seg]).to(torch.float32)
            count = torch.zeros(len(chunk), device=x.device, dtype=torch.float32)
            count = count.index_add(0, move_seg, better)[has_target]
            pol_hits += int((count < 1).sum())
            pol_hits3 += int((count < 3).sum())
            pol_hits5 += int((count < 5).sum())
            pol_n += int(has_target.sum())
            # What picking uniformly at random among the legal moves would score.
            n_legal = (corpus.move_ptr.index_select(0, chunk + 1)
                       - corpus.move_ptr.index_select(0, chunk))[has_target]
            uniform += float((1.0 / n_legal.clamp(min=1)).sum())
    pred = torch.cat(preds)
    t = corpus.y.index_select(0, idx)
    return (bce(pred, t), decisive_accuracy(pred, t),
            pol_loss / max(1, pol_n), pol_hits / max(1, pol_n),
            uniform / max(1, pol_n),
            pol_hits3 / max(1, pol_n), pol_hits5 / max(1, pol_n))


# ---------------------------------------------------------------- training

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collections", nargs="*")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--conv-type", default="GIN", choices=["GIN", "GAT", "GCN"])
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--pooling", default="add", choices=["mean", "max", "add", "concat"])
    parser.add_argument("--residual", action="store_true",
                        help="residual connections between conv layers. Off by default "
                             "to match earlier runs; worth turning on past ~4 layers, "
                             "where plain message passing starts to over-smooth")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--limit-games", type=int, default=0, help="unsupported here")
    parser.add_argument("--limit-fraction", type=float, default=0.0,
                        help="train on this fraction of the games, for smoke tests")
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="weight of the policy cross entropy in the joint loss")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-device", default=None,
                        help="where the corpus lives; defaults to the compute device")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_device = torch.device(args.data_device or device)

    amp_dtype = None
    if not args.no_amp and device.type == "cuda":
        # torch.cuda.is_bf16_supported() answers True on Turing too, because it counts
        # software emulation. Emulated bf16 is slower than fp16, so gate on the compute
        # capability instead: bf16 is native from Ampere (8.0) on. The RTX 2080 Ti is
        # 7.5 and gets fp16; the L40 is 8.9 and gets bf16.
        major = torch.cuda.get_device_capability(device)[0]
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16

    os.makedirs(args.out, exist_ok=True)
    print(f"device {device}  amp {amp_dtype}")

    corpus = Corpus(args.shards, args.collections, data_device, args.limit_games)
    splits = split_by_game(corpus.game_key, args.seed)
    n_games = len(torch.unique(corpus.game_key))
    print(f"corpus: {len(corpus)} positions from {n_games} games "
          f"in {len(corpus.collections)} collections, {corpus.bytes() / 1e9:.2f} GB")
    print("split by game: " + ", ".join(
        f"{k} {len(v)} positions / {len(torch.unique(corpus.game_key[v]))} games"
        for k, v in splits.items()))

    # The baselines the network has to beat on the validation split.
    val_idx = splits["val"].to(device)
    val_target = corpus.y.index_select(0, val_idx)
    heur = corpus.heuristic.to(val_target.device).index_select(0, val_idx)
    base_heur = bce(heur, val_target)
    base_const = bce(torch.full_like(val_target, float(val_target.mean())), val_target)
    print(f"baseline val BCE: constant {base_const:.4f}  heuristic {base_heur:.4f} "
          f"(acc {decisive_accuracy(heur, val_target):.3f})")

    model = AgentNet(
        in_dim=corpus.num_features, hidden_dim=args.hidden_dim,
        conv_type=args.conv_type, num_layers=args.num_layers, gat_heads=args.gat_heads,
        conv_dropout=args.dropout, mlp_dropout=args.dropout, final_dropout=args.dropout,
        use_layer_norm=True, use_residual=args.residual, pooling=args.pooling,
        mlp_layers=2, final_mlp_layers=2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.conv_type} x{args.num_layers}, hidden {args.hidden_dim}, "
          f"{'residual' if args.residual else 'plain'}, {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    train_idx = splits["train"].to(device)
    best = {"val_bce": float("inf"), "epoch": -1}
    log_path = os.path.join(args.out, "train_log.jsonl")
    log = open(log_path, "a")
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_idx), generator=generator).to(device)
        total, seen = 0.0, 0
        t0 = time.perf_counter()
        for start in range(0, len(perm), args.batch_size):
            chunk = train_idx.index_select(0, perm[start:start + args.batch_size])
            (x, edge_index, batch_vec, target, src, bug, dst,
             move_seg, target_pi, has_target) = corpus.batch(chunk)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                value, move_logits = model(x, edge_index, batch_vec, src, bug, dst, move_seg)
                loss = F.binary_cross_entropy_with_logits(value.float(), target)
                if has_target.any():
                    log_p = segment_log_softmax(move_logits.float(), move_seg, len(chunk))
                    # Soft cross entropy: -sum_a pi(a) log p(a), per position. With a
                    # one-hot target this is the old indexed form; with the visit
                    # distribution from self-play it is what lets the head learn the
                    # search's preferences rather than a single move.
                    per_move = -target_pi * log_p
                    per_pos = torch.zeros(len(chunk), device=x.device, dtype=per_move.dtype)
                    per_pos = per_pos.index_add(0, move_seg, per_move)
                    policy_loss = per_pos[has_target].mean()
                    loss = loss + args.policy_weight * policy_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(chunk)
            seen += len(chunk)
        scheduler.step()

        val_bce, val_acc, val_pol, val_top1, val_unif, val_t3, val_t5 = evaluate(
            model, corpus, val_idx, args.batch_size, amp_dtype)
        record = {"epoch": epoch, "train_loss": total / max(1, seen), "val_bce": val_bce,
                  "val_acc": val_acc, "val_policy_nll": val_pol, "val_top1": val_top1,
                  "val_top3": val_t3, "val_top5": val_t5,
                  "lr": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - t0}
        log.write(json.dumps(record) + "\n")
        log.flush()
        # Selection is on the combined objective: a checkpoint that predicts values well
        # but ranks moves badly is not the one the search wants.
        combined = val_bce + args.policy_weight * val_pol
        marker = ""
        if combined < best["val_bce"]:
            best = {"val_bce": combined, "value_bce": val_bce, "val_acc": val_acc,
                    "policy_nll": val_pol, "top1": val_top1, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            marker = "  <- best"
        print(f"epoch {epoch:3d}  train {record['train_loss']:.4f}  "
              f"val {val_bce:.4f} acc {val_acc:.3f}  "
              f"policy {val_pol:.4f} top1/3/5 {val_top1:.3f}/{val_t3:.3f}/{val_t5:.3f}  "
              f"{record['seconds']:.1f}s{marker}", flush=True)

        if epoch - best["epoch"] >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    # Report the held-out split with the best checkpoint, once, at the end.
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"),
                                     map_location=device, weights_only=True))
    test_idx = splits["test"].to(device)
    test_bce, test_acc, test_pol, test_top1, test_unif, test_t3, test_t5 = evaluate(
        model, corpus, test_idx, args.batch_size, amp_dtype)
    test_target = corpus.y.index_select(0, test_idx)
    test_heur = corpus.heuristic.to(test_target.device).index_select(0, test_idx)

    summary = {
        "best_epoch": best["epoch"], "val_bce": best.get("value_bce"),
        "val_combined": best["val_bce"], "val_policy_nll": best.get("policy_nll"),
        "val_top1": best.get("top1"),
        "test_bce": test_bce, "test_acc": test_acc,
        "test_policy_nll": test_pol, "test_top1": test_top1,
        "test_top3": test_t3, "test_top5": test_t5,
        "test_top1_uniform": test_unif,
        "test_bce_heuristic": bce(test_heur, test_target),
        "test_acc_heuristic": decisive_accuracy(test_heur, test_target),
        "params": n_params, "args": vars(args), "collections": corpus.collections,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    log.close()

    print(f"\nbest epoch {best['epoch']}")
    print(f"test  value     BCE {test_bce:.4f}  acc {test_acc:.3f}")
    print(f"test  policy    NLL {test_pol:.4f}  "
          f"top1 {test_top1:.3f}  top3 {test_t3:.3f}  top5 {test_t5:.3f}   "
          f"(caso: top1 {test_unif:.3f}, {test_top1 / max(1e-9, test_unif):.0f}x)")
    print(f"test  heuristic BCE {summary['test_bce_heuristic']:.4f}  "
          f"acc {summary['test_acc_heuristic']:.3f}")
    print("the network is only useful to the search if it beats the heuristic here")
    print(f"weights: {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
