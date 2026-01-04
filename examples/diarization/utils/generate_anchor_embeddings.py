#!/usr/bin/env python3
"""Generate numeric anchor token embeddings from a pretrained LLM.

This script extracts embeddings for digit tokens (0-9) from a pretrained model
and saves them to a .pt file. These embeddings are used as numeric anchors in
the diarization model to mark speaker positions.

Usage:
    python generate_anchor_embeddings.py \
        --model_path /path/to/Qwen2.5-7B-Instruct \
        --output_path digit_token_embeddings.pt
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def load_digit_token_ids(tokenizer) -> Dict[str, int]:
    """Extract token IDs for digits 0-9 from tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        Dictionary mapping digit strings to token IDs
    """
    digits = [str(i) for i in range(10)]
    token_ids = tokenizer.convert_tokens_to_ids(digits)
    token_map = dict(zip(digits, token_ids))
    return token_map


def load_embedding_matrix(model_path: str, weight_name: str = "model.embed_tokens.weight"):
    """Load embedding matrix from a safetensors sharded model.
    
    Args:
        model_path: Path to model directory containing safetensors files
        weight_name: Name of the embedding weight tensor
        
    Returns:
        Embedding matrix tensor of shape [vocab_size, hidden_dim]
        
    Raises:
        ValueError: If weight_name not found in model index
        FileNotFoundError: If model files are missing
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Model index file not found: {index_path}\n"
            f"Make sure the model is saved in safetensors format."
        )
    
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    if weight_name not in weight_map:
        available_keys = list(weight_map.keys())[:10]
        raise ValueError(
            f"Weight '{weight_name}' not found in model index.\n"
            f"Available keys (first 10): {available_keys}..."
        )

    shard_file = weight_map[weight_name]
    shard_path = os.path.join(model_path, shard_file)
    
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file not found: {shard_path}")

    with safe_open(shard_path, framework="pt") as f:
        embedding_matrix = f.get_tensor(weight_name)

    return embedding_matrix


def generate_anchor_embeddings(
    model_path: str,
    output_path: str,
    dtype: str = "float16",
    weight_name: str = "model.embed_tokens.weight",
) -> Dict[str, any]:
    """Generate and save numeric anchor embeddings.
    
    Args:
        model_path: Path to pretrained LLM (e.g., Qwen2.5-7B-Instruct)
        output_path: Output .pt file path
        dtype: Data type for saved embeddings ("float16" or "float32")
        weight_name: Name of embedding weight in model
        
    Returns:
        Dictionary containing tokens, token_ids, embeddings, and metadata
    """
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    print("Extracting digit token IDs...")
    digit_token_map = load_digit_token_ids(tokenizer)
    digit_tokens: List[str] = list(digit_token_map.keys())
    digit_ids: List[int] = [digit_token_map[d] for d in digit_tokens]
    
    print(f"Loading embedding matrix from: {model_path}")
    embedding_matrix = load_embedding_matrix(model_path, weight_name)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    # Extract embeddings for digit tokens
    digit_embeddings = embedding_matrix[digit_ids]
    
    # Convert to specified dtype
    if dtype == "float32":
        digit_embeddings = digit_embeddings.to(torch.float32)
    else:
        digit_embeddings = digit_embeddings.to(torch.float16)
    
    # Package embeddings with metadata
    payload = {
        "tokens": digit_tokens,
        "token_ids": digit_ids,
        "embeddings": digit_embeddings,
        "dtype": str(digit_embeddings.dtype),
        "shape": tuple(digit_embeddings.shape),
        "model_path": model_path,
    }
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(payload, output_path)
    
    print("\n" + "="*60)
    print("Successfully saved numeric anchor embeddings:")
    print(f"  Tokens: {digit_tokens}")
    print(f"  Token IDs: {digit_ids}")
    print(f"  Shape: {tuple(digit_embeddings.shape)}")
    print(f"  Dtype: {digit_embeddings.dtype}")
    print(f"  Output file: {output_path}")
    print("="*60)
    
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Generate numeric anchor token embeddings from pretrained LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings from Qwen2.5-7B
  python generate_anchor_embeddings.py \\
      --model_path /path/to/Qwen2.5-7B-Instruct \\
      --output_path digit_token_embeddings.pt
  
  # Use float32 precision
  python generate_anchor_embeddings.py \\
      --model_path /path/to/Qwen2.5-7B-Instruct \\
      --output_path digit_token_embeddings.pt \\
      --dtype float32
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained LLM model directory (e.g., Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="digit_token_embeddings.pt",
        help="Output .pt file path (default: digit_token_embeddings.pt)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for saved embeddings (default: float16)",
    )
    parser.add_argument(
        "--weight_name",
        type=str,
        default="model.embed_tokens.weight",
        help="Name of embedding weight tensor in model (default: model.embed_tokens.weight)",
    )
    
    args = parser.parse_args()
    
    try:
        generate_anchor_embeddings(
            model_path=args.model_path,
            output_path=args.output_path,
            dtype=args.dtype,
            weight_name=args.weight_name,
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

