"""Train the value network on the rebuilt shards.

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

from ai.graph_network import GraphClassifier  # noqa: E402


# ---------------------------------------------------------------- data

class Corpus:
    """All shards concatenated, pre-collated, optionally resident on the device."""

    def __init__(self, shard_dir, collections=None, device="cpu", limit_games=0):
        xs, edges, ys, heur, game_keys, node_counts, edge_counts = [], [], [], [], [], [], []
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
        np.cumsum(np.concatenate(node_counts), out=node_ptr[1:])
        np.cumsum(np.concatenate(edge_counts), out=edge_ptr[1:])

        if limit_games:
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

    def __len__(self):
        return len(self.y)

    def bytes(self):
        return (self.x.numel() * 4 + self.edge_index.numel() * 8
                + self.node_ptr.numel() * 8 + self.edge_ptr.numel() * 8)

    def batch(self, idx):
        """Gather a batch of graphs into (x, edge_index, batch_vector, y)."""
        node_index, node_seg, new_node_ptr = _gather(self.node_ptr, idx)
        edge_pos, edge_seg, _ = _gather(self.edge_ptr, idx)
        x = self.x.index_select(0, node_index)
        # Stored edges are local to their own graph, so they only need the new offset.
        edge_index = self.edge_index.index_select(1, edge_pos) + new_node_ptr[:-1][edge_seg]
        return x, edge_index, node_seg, self.y.index_select(0, idx)


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
    preds = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        x, edge_index, batch_vec, _ = corpus.batch(chunk)
        with torch.autocast(corpus.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model.model(x, edge_index, batch_vec).squeeze(-1)
        preds.append(torch.sigmoid(logits.float()))
    pred = torch.cat(preds)
    target = corpus.y.index_select(0, idx)
    return bce(pred, target), decisive_accuracy(pred, target)


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
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--limit-games", type=int, default=0, help="smoke-test on N games")
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

    model = GraphClassifier(
        in_dim=corpus.num_features, hidden_dim=args.hidden_dim, num_classes=1,
        lr=args.lr, weight_decay=args.weight_decay,
        conv_type=args.conv_type, num_layers=args.num_layers, gat_heads=args.gat_heads,
        conv_dropout=args.dropout, mlp_dropout=args.dropout, final_dropout=args.dropout,
        use_layer_norm=True, use_residual=False, pooling=args.pooling,
        mlp_layers=2, final_mlp_layers=2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.conv_type} x{args.num_layers}, hidden {args.hidden_dim}, "
          f"{n_params / 1e6:.2f}M parameters")

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
            x, edge_index, batch_vec, target = corpus.batch(chunk)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model.model(x, edge_index, batch_vec).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(chunk)
            seen += len(chunk)
        scheduler.step()

        val_bce, val_acc = evaluate(model, corpus, val_idx, args.batch_size, amp_dtype)
        record = {"epoch": epoch, "train_bce": total / max(1, seen), "val_bce": val_bce,
                  "val_acc": val_acc, "lr": scheduler.get_last_lr()[0],
                  "seconds": time.perf_counter() - t0}
        log.write(json.dumps(record) + "\n")
        log.flush()
        marker = ""
        if val_bce < best["val_bce"]:
            best = {"val_bce": val_bce, "val_acc": val_acc, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            marker = "  <- best"
        print(f"epoch {epoch:3d}  train {record['train_bce']:.4f}  val {val_bce:.4f}  "
              f"acc {val_acc:.3f}  {record['seconds']:.1f}s{marker}", flush=True)

        if epoch - best["epoch"] >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    # Report the held-out split with the best checkpoint, once, at the end.
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"),
                                     map_location=device, weights_only=True))
    test_idx = splits["test"].to(device)
    test_bce, test_acc = evaluate(model, corpus, test_idx, args.batch_size, amp_dtype)
    test_target = corpus.y.index_select(0, test_idx)
    test_heur = corpus.heuristic.to(test_target.device).index_select(0, test_idx)

    summary = {
        "best_epoch": best["epoch"], "val_bce": best["val_bce"],
        "test_bce": test_bce, "test_acc": test_acc,
        "test_bce_heuristic": bce(test_heur, test_target),
        "test_acc_heuristic": decisive_accuracy(test_heur, test_target),
        "params": n_params, "args": vars(args), "collections": corpus.collections,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    log.close()

    print(f"\nbest epoch {best['epoch']}  val BCE {best['val_bce']:.4f}")
    print(f"test  network   BCE {test_bce:.4f}  acc {test_acc:.3f}")
    print(f"test  heuristic BCE {summary['test_bce_heuristic']:.4f}  "
          f"acc {summary['test_acc_heuristic']:.3f}")
    print("the network is only useful to the search if it beats the heuristic here")
    print(f"weights: {os.path.join(args.out, 'best.pt')}")


if __name__ == "__main__":
    main()
