#!/usr/bin/env python3
"""
Quick inference for the Sequential HIVE LLM.

Loads the best checkpoint, pulls one sample from the validation split,
and prints the predicted legal move, score, and optional text generation.
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from A_LLM_trainer import (
    SequentialHiveLLMConfig,
    SequentialHiveLLMPlayer,
    SequentialHiveDataset,
)


def load_config(config_path: str) -> SequentialHiveLLMConfig:
    with open(config_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    cfg = SequentialHiveLLMConfig()
    # Set attributes from saved config where available
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    model_dir = os.path.join("models", "sequential_hive_llm_v1", "best_model")
    assert os.path.isdir(model_dir), f"Best model not found at {model_dir}"

    # Load config saved with the checkpoint
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = load_config(cfg_path)

    # Init player/model (this loads base model + tokenizer + LoRA skeleton)
    player = SequentialHiveLLMPlayer(cfg)

    # Load trained LoRA adapter weights under a distinct name and activate
    adapter_path = os.path.join(model_dir, "lora_adapter")
    try:
        player.base_model.load_adapter(adapter_path, adapter_name="trained")
        player.base_model.set_adapter("trained")
    except Exception as e:
        print(f"Warning: failed to load adapter from {adapter_path}: {e}")

    # Projection bridge (AE) is internal to the player and already loaded; nothing to load here.
    device = next(player.base_model.parameters()).device
    player.projection_module.to(device)
    player.eval()

    # Load a validation sample
    data_dir = os.path.join("data", "sequential_hive_llm_dataset")
    val_ds = SequentialHiveDataset(data_dir, split="validation", config=cfg)
    assert len(val_ds) > 0, "Validation dataset is empty"
    sample = val_ds[0]

    # Prepare tensors with batch dim = 1
    def as_batch(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 3 else x.unsqueeze(0)

    context_board_embeddings = as_batch(sample["context_board_embeddings"]).to(device)
    context_move_embeddings = as_batch(sample["context_move_embeddings"]).to(device)
    current_board_embedding = as_batch(sample["current_board_embedding"]).to(device)
    legal_move_embeddings = as_batch(sample["legal_move_embeddings"]).to(device)

    # Build legal move mask like in collate_fn
    num_moves = sample["legal_move_embeddings"].shape[0]
    legal_move_masks = torch.zeros(num_moves, dtype=torch.bool)
    legal_move_masks[:num_moves] = True
    legal_move_masks = legal_move_masks.unsqueeze(0).to(device)

    # Monkey-patch the move similarity to silence verbose prints
    def _silent_find(predicted_move: torch.Tensor, legal_moves: torch.Tensor, legal_move_mask: torch.Tensor, projection_module) -> Tuple[int, float]:
        with torch.no_grad():
            proj = projection_module.move_projection(legal_moves.unsqueeze(0)).squeeze(0)  # [N, D]
            diffs = proj - predicted_move.unsqueeze(0)
            mse = (diffs.pow(2).mean(dim=1))
            scores = -mse
            scores = scores.masked_fill(~legal_move_mask, -float("inf"))
            idx = scores.argmax().item()
            return idx, float(scores[idx].item())
    player.move_similarity.find_closest_move = _silent_find  # type: ignore

    # Run move generation (embedding-space)
    with torch.no_grad():
        best_idx, sim, predicted_move, predicted_state = player.generate_move(
            context_board_embeddings,
            context_move_embeddings,
            current_board_embedding,
            legal_move_embeddings,
            legal_move_masks,
        )

    chosen_idx = int(sample["chosen_move_idx"]) if isinstance(sample["chosen_move_idx"], (int,)) else int(sample["chosen_move_idx"])  # type: ignore
    legal_texts = sample.get("legal_move_texts", [])
    pred_text = legal_texts[best_idx] if 0 <= best_idx < len(legal_texts) else "<UNK>"
    chosen_text = legal_texts[chosen_idx] if 0 <= chosen_idx < len(legal_texts) else "<UNK>"

    print("=== Sequential HIVE LLM Inference ===")
    print(f"Best move idx: {best_idx} | score: {sim:.4f}")
    print(f"Predicted move text: {pred_text}")
    print(f"Chosen (gt) move idx: {chosen_idx} | text: {chosen_text}")
    print(f"Match: {best_idx == chosen_idx}")

    # Show the human-readable prompt preview consistent with the trainer
    def render_prompt_preview(seq_len: int) -> str:
        parts: List[str] = []
        if getattr(player.tokenizer, 'bos_token_id', None) is not None:
            parts.append("[BOS]")
        parts.append(
            "You are playing the board game HIVE. These are the previous board states and moves made by the players that alternate playing:\n"
        )
        for i in range(seq_len):
            parts.append(f"Board {i}: [STATE_{i}] Move {i}: [MOVE_{i}]\n" if i < seq_len - 1 else f"Board {i}: [STATE_{i}] Move {i}: [MOVE_{i}]")
        parts.append("\n\nPredict the next board state and move. RESPECT THE FOLLOWING OUTPUT FORMAT:\n")
        parts.append(f"Board {seq_len}: <BOARD> Move {seq_len}: <MOVE>")
        return "".join(parts)

    hist_len = context_board_embeddings.shape[1]
    print("\n=== Prompt preview ===")
    print(render_prompt_preview(hist_len))

    # Show top-5 legal moves by cosine similarity (GIN space)
    with torch.no_grad():
        pred_move_gin = player.projection_module.move_projection_inverse(predicted_move)  # [gin_dim]
        legal_moves_gin = legal_move_embeddings.squeeze(0).squeeze(1)  # [M, gin_dim]
        sims = torch.nn.functional.cosine_similarity(
            pred_move_gin.unsqueeze(0), legal_moves_gin, dim=1
        )
        sims_masked = sims.masked_fill(~legal_move_masks.squeeze(0), -float("inf"))
        topk = min(5, sims_masked.numel())
        vals, idxs = torch.topk(sims_masked, k=topk)
        print("\nTop-5 candidates:")
        for rank, (v, i) in enumerate(zip(vals.tolist(), idxs.tolist()), 1):
            text_i = legal_texts[i] if 0 <= i < len(legal_texts) else "<UNK>"
            print(f"{rank}. idx={i:>3} sim={v:6.4f} text={text_i}")

    # Optional: try textual generation to inspect if it continues the prompt
    try:
        input_embeds, attention_mask, board_pos, move_pos = player._create_sequential_soft_prompt(
            context_board_embeddings,
            context_move_embeddings,
            current_board_embedding,
            legal_move_embeddings,
            legal_move_masks,
        )
        # Print the positions used for <BOARD> and <MOVE> within the built sequence
        print("\nPositions:")
        print(f"board_pos (index of <BOARD> token in sequence): {board_pos.tolist()}")
        print(f"move_pos  (index of <MOVE>  token in sequence): {move_pos.tolist()}")
        gen_ids = player.base_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            max_new_tokens=64,
            min_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=player.tokenizer.pad_token_id,
            # Let it run for min_new_tokens even if it wants to emit EOS early
            eos_token_id=None,
        )
        # With inputs_embeds, many models return only generated tokens. Decode directly.
        text = player.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        print("\n=== Generated text continuation ===")
        print(text)
    except Exception as e:
        print(f"Text generation failed (non-fatal): {e}")

    # Quick accuracy over a few validation samples
    try:
        n_eval = min(64, len(val_ds))
        correct = 0
        sims_accum: List[float] = []
        for idx in range(n_eval):
            s = val_ds[idx]
            cb = s["context_board_embeddings"].unsqueeze(0).to(device)
            cm = s["context_move_embeddings"].unsqueeze(0).to(device)
            cur = s["current_board_embedding"].unsqueeze(0).to(device)
            lm = s["legal_move_embeddings"].unsqueeze(0).to(device)
            nm = s["legal_move_embeddings"].shape[0]
            msk = torch.zeros(nm, dtype=torch.bool)
            msk[:nm] = True
            msk = msk.unsqueeze(0).to(device)
            # Use the same silent similarity
            # Forward once to get predicted_move at the recorded move_pos
            inp, am, bpos, mpos = player._create_sequential_soft_prompt(cb, cm, cur, lm, msk)
            out = player.base_model(inputs_embeds=inp, attention_mask=am, output_hidden_states=True, return_dict=True)
            last = out.hidden_states[-1]
            pred_move = last[0, mpos.item(), :]
            pred_move_gin = player.projection_module.move_projection_inverse(pred_move)
            lm_gin = lm.squeeze(0).squeeze(1)
            sims = F.cosine_similarity(pred_move_gin.unsqueeze(0), lm_gin, dim=1)
            sims = sims.masked_fill(~msk.squeeze(0), -float("inf"))
            bidx = int(sims.argmax().item())
            bsim = float(sims[bidx].item())
            gt = int(s["chosen_move_idx"]) if isinstance(s["chosen_move_idx"], (int,)) else int(s["chosen_move_idx"])  # type: ignore
            correct += int(bidx == gt)
            sims_accum.append(float(bsim))
        acc = correct / max(1, n_eval)
        mean_sim = sum(sims_accum) / max(1, len(sims_accum))
        print("\n=== Quick validation probe ===")
        print(f"Samples: {n_eval} | Top-1 accuracy: {acc:.3f} | Mean sim: {mean_sim:.4f}")
    except Exception as e:
        print(f"Validation probe failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
