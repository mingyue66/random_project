"""SOT (Serialized Output Training) ASR training script.

This script extends the standard ASR training to support Speaker-aware Output Training
where the model learns to predict speaker change tokens (<sc>) along with text.

Key differences from standard ASR training:
- Adds <sc> token to tokenizer as a special token
- Handles vocab size expansion when loading pretrained models
- Expects data with speaker turn information (text contains <sc> markers)
- Provides debug logging for <sc> token handling

Typical usage:
    torchrun --nproc_per_node=1 examples/asr/train_sot.py --config-name train_sot

Config expectations (in addition to standard ASR config):
- ``cfg.sot_training``: bool, whether to enable SOT training (default: True)
- ``cfg.model.pretrained_model``: optional path to pretrained ASR model to finetune
"""

import json
import logging
import os

import hydra
import torch
import torch.distributed as dist
from data_module import AsrDatamodule
from lhotse.utils import fix_random_seed
from omegaconf import DictConfig, OmegaConf
from trainer_sot import AsrSotTrainer
from transformers import AutoTokenizer

from framework.auto.auto_config import AutoConfig
from framework.auto.auto_model import AutoModel
from framework.models.zipformer.utils.scaling import ScaledLinear


def load_pretrained_encoder(cfg: DictConfig):
    """Load pretrained encoder module or build an empty encoder config."""
    if cfg.get("model_type") == "zipformer":
        from framework.models.zipformer.model import ZipformerEncoderModel
        from framework.models.zipformer.model_config import ZipformerConfig

        if cfg.get("pretrained_encoder") is not None:
            pretrained_encoder = ZipformerEncoderModel.from_pretrained(
                cfg.pretrained_encoder
            )
            encoder_config = pretrained_encoder.config
            return encoder_config, pretrained_encoder
        else:
            encoder_config = ZipformerConfig()
            return encoder_config, None
    else:
        raise ValueError(f"Unsupported encoder model type: {cfg.get('model_type')}")


@hydra.main(version_base=None, config_path="configs", config_name="train_sot")
def main(cfg: DictConfig):
    """Hydra entrypoint for SOT ASR training.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra/OMEGACONF configuration. Expected keys include:
        - ``seed``: random seed.
        - ``exp_dir``: experiment directory to save artifacts.
        - ``sot_training``: whether to enable SOT training (default: True).
        - ``model``: contains ``model_type``, ``encoder_type``, and optional ``pretrained_model``.
        - ``tokenizer``: tokenizer id or path for HF ``AutoTokenizer.from_pretrained``.
        - ``data``: datamodule configuration (see ``AsrDatamodule`` and base datamodule).

    Side Effects
    ------------
    - Initializes/disposes torch.distributed process group when WORLD_SIZE > 1.
    - Saves model config and tokenizer on rank 0 under ``exp_dir``.
    - Starts the training loop via ``AsrSotTrainer.run()``.
    """
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    # 1) Fix random seed
    if "seed" in cfg:
        fix_random_seed(cfg.seed)

    # 2) Gather torchrun environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 3) Initialize process group if multi-GPU
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    # 4) Create experiment directory
    if "exp_dir" in cfg and cfg.exp_dir:
        os.makedirs(cfg.exp_dir, exist_ok=True)

    # 5) Check if loading from pretrained ASR model
    pretrained_model_path = cfg.model.get("pretrained_model", None)
    sot_training = cfg.get("sot_training", True)
    
    if pretrained_model_path:
        # Load pretrained model and expand vocab for <sc> token
        if rank == 0:
            logging.info("=" * 80)
            logging.info(f"Loading pretrained ASR model from: {pretrained_model_path}")
            logging.info(f"SOT training enabled: {sot_training}")
            logging.info("=" * 80)
        
        # Step 1: Load config and tokenizer manually (to handle Framework tokenizer compatibility)
        from pathlib import Path
        model_dir = Path(pretrained_model_path)
        
        # Load config
        config = AutoConfig.from_pretrained(model_dir)
        
        # Try to load tokenizer from tokenizer subdirectory first (for old Framework models)
        tokenizer_dir = model_dir / "tokenizer"
        if tokenizer_dir.exists():
            if rank == 0:
                logging.info(f"Loading tokenizer from: {tokenizer_dir}")
            try:
                # Try Framework tokenizer first
                from framework.auto.auto_tokenizer import AutoTokenizer as FrameworkAutoTokenizer
                tokenizer = FrameworkAutoTokenizer.from_pretrained(str(tokenizer_dir))
                if rank == 0:
                    logging.info("Loaded Framework tokenizer, will convert to HF tokenizer")
                
                # Convert Framework tokenizer to HF tokenizer by creating a new one with same vocab
                # For simplicity, we'll load it as-is and add <sc> later
            except Exception as e:
                if rank == 0:
                    logging.warning(f"Failed to load Framework tokenizer: {e}")
                    logging.info("Trying HF tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        else:
            # Load from model_dir directly
            if rank == 0:
                logging.info(f"Loading tokenizer from: {model_dir}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Step 2: Build model with config and tokenizer
        from framework.models.asr.model import AsrModel
        model = AsrModel(config, tokenizer)
        
        # Step 3: Load weights
        weight_path = None
        for ext in (".safetensors", ".pt"):
            for name in ("pretrained", "model"):
                p = model_dir / f"{name}{ext}"
                if p.exists():
                    weight_path = p
                    break
            if weight_path is not None:
                break
        
        if weight_path is None:
            raise FileNotFoundError(
                f"Expected one of ['pretrained.safetensors','model.safetensors','pretrained.pt','model.pt'] under {model_dir}"
            )
        
        if rank == 0:
            logging.info(f"Loading weights from: {weight_path}")
        
        ext = weight_path.suffix.lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file as safe_load_file
            state_dict = safe_load_file(str(weight_path), device="cpu")
        else:
            state_obj = torch.load(str(weight_path), map_location="cpu")
            state_dict = (
                state_obj["state_dict"]
                if isinstance(state_obj, dict) and "state_dict" in state_obj
                else state_obj
            )
        
        model.load_state_dict(state_dict, strict=False)
        
        original_vocab_size = len(tokenizer)
        if rank == 0:
            logging.info(f"Original vocab size: {original_vocab_size}")
        
        # Step 2: Add <sc> token if SOT training is enabled
        if sot_training:
            if rank == 0:
                logging.info("Adding <sc> token for SOT training...")
            
            # Add <sc> as a special token to the tokenizer
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<sc>"]},
                replace_additional_special_tokens=False
            )
            
            new_vocab_size = len(tokenizer)
            
            if rank == 0:
                logging.info(f"Added {num_added} special token(s)")
                logging.info(f"New vocab size: {new_vocab_size}")
            
            # Store <sc> token ID for convenience
            tokenizer.sc_id = tokenizer.convert_tokens_to_ids("<sc>")
            tokenizer.sot_training = True
            
            # Step 3: Resize model embeddings to accommodate new token
            if num_added > 0:
                if rank == 0:
                    logging.info("Resizing model embeddings for new vocabulary...")
                
                # Get the old vocab size from model
                old_vocab_size = model.vocab_size
                
                # Resize decoder embedding
                old_embed = model.decoder.embedding.weight.data
                new_embed = torch.nn.Embedding(new_vocab_size, old_embed.shape[1])
                new_embed.weight.data[:old_vocab_size] = old_embed
                # Initialize new token embedding with small random values
                torch.nn.init.normal_(new_embed.weight.data[old_vocab_size:], mean=0.0, std=0.02)
                model.decoder.embedding = new_embed
                
                # Resize joiner output layer
                old_output = model.joiner.output_linear.weight.data
                new_output_linear = ScaledLinear(
                    old_output.shape[1], 
                    new_vocab_size, 
                    bias=True
                )
                new_output_linear.weight.data[:old_vocab_size] = old_output
                torch.nn.init.normal_(new_output_linear.weight.data[old_vocab_size:], mean=0.0, std=0.02)
                if model.joiner.output_linear.bias is not None:
                    new_output_linear.bias.data[:old_vocab_size] = model.joiner.output_linear.bias.data
                model.joiner.output_linear = new_output_linear
                
                # Resize simple_am_proj if exists
                if hasattr(model, 'simple_am_proj') and model.simple_am_proj is not None:
                    old_am = model.simple_am_proj.weight.data
                    new_am_proj = ScaledLinear(old_am.shape[1], new_vocab_size, bias=True)
                    new_am_proj.weight.data[:old_vocab_size] = old_am
                    torch.nn.init.normal_(new_am_proj.weight.data[old_vocab_size:], mean=0.0, std=0.02)
                    if model.simple_am_proj.bias is not None:
                        new_am_proj.bias.data[:old_vocab_size] = model.simple_am_proj.bias.data
                    model.simple_am_proj = new_am_proj
                
                # Resize simple_lm_proj if exists
                if hasattr(model, 'simple_lm_proj') and model.simple_lm_proj is not None:
                    old_lm = model.simple_lm_proj.weight.data
                    new_lm_proj = ScaledLinear(old_lm.shape[1], new_vocab_size, bias=True)
                    new_lm_proj.weight.data[:old_vocab_size] = old_lm
                    torch.nn.init.normal_(new_lm_proj.weight.data[old_vocab_size:], mean=0.0, std=0.02)
                    if model.simple_lm_proj.bias is not None:
                        new_lm_proj.bias.data[:old_vocab_size] = model.simple_lm_proj.bias.data
                    model.simple_lm_proj = new_lm_proj
                
                # Update model's vocab_size attribute
                model.vocab_size = new_vocab_size
                
                if rank == 0:
                    logging.info(f"Model embeddings resized from {old_vocab_size} to {new_vocab_size}")
        
        if rank == 0:
            num_params = sum(p.numel() for p in model.parameters()) / 1e6
            logging.info(f"Loaded pretrained model; params={num_params:.2f} M")
            logging.info(f"Final vocab size: {model.vocab_size}")
            if hasattr(tokenizer, 'sc_id'):
                logging.info(f"<sc> token ID: {tokenizer.sc_id}")
    else:
        # Build model from scratch
        encoder_config, pretrained_encoder = load_pretrained_encoder(cfg.model.encoder)
        config = AutoConfig.for_model(cfg.model.model_type, encoder_config=encoder_config)
        
        # Load tokenizer and add <sc> token if SOT training
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        
        if sot_training:
            if rank == 0:
                logging.info("=" * 80)
                logging.info("Adding <sc> token for SOT training")
                logging.info(f"Original vocab size: {tokenizer.vocab_size}")
            
            # Add <sc> as a special token
            num_added = tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<sc>"]},
                replace_additional_special_tokens=False
            )
            
            if rank == 0:
                logging.info(f"Added {num_added} special token(s)")
                logging.info(f"New vocab size: {tokenizer.vocab_size}")
                logging.info("=" * 80)
            
            # Store <sc> token ID for convenience
            tokenizer.sc_id = tokenizer.convert_tokens_to_ids("<sc>")
            tokenizer.sot_training = True
        
        # Build model with tokenizer (vocab_size will reflect the added token)
        model = AutoModel.from_config(config, tokenizer)
        
        # Load pretrained encoder if provided
        if pretrained_encoder is not None:
            model.encoder.load_state_dict(pretrained_encoder.state_dict(), strict=True)
            if rank == 0:
                logging.info(
                    f"Loaded pretrained encoder from {cfg.model.encoder.pretrained_encoder}"
                )

    # Debug: Show tokenizer and <sc> token info
    if rank == 0 and sot_training:
        logging.info("=" * 80)
        logging.info("Tokenizer and SOT Configuration:")
        logging.info(f"  Tokenizer type: {type(tokenizer).__name__}")
        logging.info(f"  Vocab size: {tokenizer.vocab_size}")
        logging.info(f"  SOT training: {getattr(tokenizer, 'sot_training', False)}")
        if hasattr(tokenizer, 'sc_id'):
            logging.info(f"  <sc> token ID: {tokenizer.sc_id}")
        
        # Test <sc> token encoding/decoding
        test_texts = [
            "你好 <sc> 世界",
            "测试 <sc> 多个 <sc> 标签",
            "没有标签的文本",
        ]
        logging.info("\n  Testing <sc> token encode/decode:")
        for test_text in test_texts:
            try:
                # Test with model's tokenizer.encode (returns List[List[int]])
                encoded = tokenizer.encode(test_text)
                # For HF tokenizer, encode might return different format
                if isinstance(encoded, list) and len(encoded) > 0:
                    if isinstance(encoded[0], list):
                        token_ids = encoded[0]
                    else:
                        token_ids = encoded
                else:
                    token_ids = encoded
                
                # Decode back
                if hasattr(tokenizer, 'decode') and callable(tokenizer.decode):
                    if isinstance(token_ids, list):
                        decoded = tokenizer.decode(token_ids)
                    else:
                        decoded = tokenizer.decode([token_ids])
                    if isinstance(decoded, list):
                        decoded = decoded[0] if decoded else ""
                else:
                    decoded = "N/A"
                
                has_sc_in_text = "<sc>" in test_text
                has_sc_in_decoded = "<sc>" in str(decoded) if decoded != "N/A" else False
                status = "✓" if (has_sc_in_text == has_sc_in_decoded) else "✗"
                
                logging.info(f"    {status} Input: '{test_text}'")
                logging.info(f"      Token IDs: {token_ids[:15]}..." if len(token_ids) > 15 else f"      Token IDs: {token_ids}")
                logging.info(f"      Decoded: '{decoded}'")
                
                if has_sc_in_text and hasattr(tokenizer, 'sc_id'):
                    sc_count_in_ids = token_ids.count(tokenizer.sc_id) if isinstance(token_ids, list) else 0
                    sc_count_in_text = test_text.count("<sc>")
                    if sc_count_in_ids != sc_count_in_text:
                        logging.warning(f"      ⚠️  <sc> count mismatch: {sc_count_in_text} in text vs {sc_count_in_ids} in IDs")
            except Exception as e:
                logging.error(f"    ✗ Error testing '{test_text}': {e}")
        
        logging.info("=" * 80)

    # 6) Save config and tokenizer
    if rank == 0:
        config.save_pretrained(cfg.exp_dir)
        tokenizer.save_pretrained(cfg.exp_dir)
        
        # Save SOT training flag
        sot_config_path = os.path.join(cfg.exp_dir, "sot_config.json")
        with open(sot_config_path, "w") as f:
            json.dump({
                "sot_training": sot_training,
                "sc_token": "<sc>",
                "sc_token_id": getattr(tokenizer, 'sc_id', None),
            }, f, indent=2)
        logging.info(f"Saved SOT config to {sot_config_path}")

    # 7) Initialize data module
    data_module = AsrDatamodule(cfg.data)

    # 8) Create the SOT trainer
    trainer = AsrSotTrainer(
        cfg, model, data_module, rank=rank, local_rank=local_rank, world_size=world_size
    )
    trainer.run()

    # 9) Destroy process group if used
    if world_size > 1:
        dist.destroy_process_group()

    logging.info("SOT ASR training finished successfully.")


if __name__ == "__main__":
    main()

