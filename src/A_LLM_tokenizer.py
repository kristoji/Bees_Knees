#!/usr/bin/env python3
import os
import sys
import argparse
from transformers import AutoTokenizer

def load_tokenizer(path_or_name: str):
    print(f"[info] Loading tokenizer from: {path_or_name}")
    tok = AutoTokenizer.from_pretrained(
        path_or_name,
        trust_remote_code=True,
        use_fast=True,
        padding_side="left",
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        print("[info] pad_token was None → set to eos_token")
    return tok

def main():
    parser = argparse.ArgumentParser(description="Tokenizer encode/decode REPL (Ctrl+C to quit)")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/hive_llm/tokenizer",
        help="Path or model name for the tokenizer (default: models/hive_llm/tokenizer)",
    )
    args = parser.parse_args()

    # If default path doesn't exist, try to treat as model name (e.g., gpt2)
    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        print(f"[warn] Path not found: {tok_path} → trying as model name")
    tokenizer = load_tokenizer(tok_path)

    print("\n[ready] Type a line to encode/decode. Press Ctrl+C to exit.\n")
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                print("\n[bye]")
                break

            # Encode
            encoded = tokenizer(line, add_special_tokens=True)
            input_ids = encoded["input_ids"]
            if isinstance(input_ids[0], list):  # batch dimension if present
                input_ids = input_ids[0]

            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            # Decode
            decoded_skip = tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_all = tokenizer.decode(input_ids, skip_special_tokens=False)

            # Print results
            print(f"IDs   ({len(input_ids)}): {input_ids}")
            print(f"Tokens({len(tokens)}): {tokens}")
            print(f"Decoded (skip_special_tokens=True):  {decoded_skip}")
            print(f"Decoded (skip_special_tokens=False): {decoded_all}")
            print()
    except KeyboardInterrupt:
        print("\n[bye]")

if __name__ == "__main__":
    main()
