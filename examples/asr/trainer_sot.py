"""SOT (Serialized Output Training) ASR Trainer.

This trainer extends the base ASR trainer to provide enhanced logging and monitoring
for SOT training, where the model learns to predict speaker change tokens (<sc>).

Key features:
- Debug logging for <sc> token presence in training batches
- Statistics tracking for <sc> token distribution
- Same training logic as base ASR trainer (no changes to loss computation)
"""

import logging
import torch

from framework.trainer.ddp_trainer import BaseTrainer
from framework.utils.metric_tracker import MetricsTracker


class AsrSotTrainer(BaseTrainer):
    """ASR trainer with SOT-specific logging and monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track <sc> token statistics
        self.sc_batch_count = 0
        self.sc_total_count = 0
        self.total_batch_count = 0
    
    def _forward_one_batch(self, batch: dict, is_training: bool, return_emb=False):
        """Forward one batch through the model.
        
        This method extends the base trainer to add SOT-specific logging.
        The actual training logic (loss computation) remains unchanged.
        """
        device = self.device
        feature = batch["inputs"]
        # at entry, feature is (N, T, C)
        assert feature.ndim == 3
        feature = feature.to(device)

        supervisions = batch["supervisions"]
        feature_lens = supervisions["num_frames"].to(device)

        batch_idx_train = self.global_step
        warm_step = self.cfg.trainer.rnnt_warm_step

        texts = batch["supervisions"]["text"]
        batch_size = len(texts)
        
        # SOT-specific logging: Track <sc> token presence in batches
        if is_training:
            self.total_batch_count += 1
            has_sc_tokenizer = (
                hasattr(self.model, 'tokenizer') and 
                hasattr(self.model.tokenizer, 'sot_training') and 
                self.model.tokenizer.sot_training
            )
            
            if has_sc_tokenizer:
                # Count utterances with <sc> token
                sc_count = sum(1 for text in texts if "<sc>" in text)
                self.sc_total_count += sc_count
                
                if sc_count > 0:
                    self.sc_batch_count += 1
                
                # Log first few batches in detail
                if self.global_step < 5 and sc_count > 0:
                    logging.info(
                        f"[SOT Batch {self.global_step}] Found {sc_count}/{batch_size} "
                        f"utterances with <sc> token"
                    )
                    
                    # Show first example with <sc>
                    for i, text in enumerate(texts):
                        if "<sc>" in text:
                            try:
                                # Count <sc> occurrences
                                num_sc = text.count("<sc>")
                                logging.info(
                                    f"  Example {i}: {num_sc} <sc> token(s) found"
                                )
                                logging.info(f"    Text preview: '{text[:100]}...'")
                                
                                # Try encoding to verify <sc> token handling
                                if hasattr(self.model.tokenizer, 'encode'):
                                    encoded = self.model.tokenizer.encode(text)
                                    if isinstance(encoded, list) and len(encoded) > 0:
                                        token_ids = encoded[0] if isinstance(encoded[0], list) else encoded
                                        logging.info(
                                            f"    Encoded length: {len(token_ids)} tokens"
                                        )
                                        
                                        if hasattr(self.model.tokenizer, 'sc_id'):
                                            sc_in_tokens = (
                                                token_ids.count(self.model.tokenizer.sc_id) 
                                                if isinstance(token_ids, list) else 0
                                            )
                                            if sc_in_tokens != num_sc:
                                                logging.warning(
                                                    f"    ⚠️  <sc> encoding mismatch: "
                                                    f"{num_sc} in text vs {sc_in_tokens} in tokens"
                                                )
                            except Exception as e:
                                logging.warning(f"  Example {i}: Error during logging: {e}")
                            break  # Only show first example
                
                # Periodic statistics (every 100 batches)
                if self.global_step > 0 and self.global_step % 100 == 0:
                    batch_rate = (
                        self.sc_batch_count / self.total_batch_count * 100 
                        if self.total_batch_count > 0 else 0
                    )
                    logging.info(
                        f"[SOT Stats @ step {self.global_step}] "
                        f"Batches with <sc>: {self.sc_batch_count}/{self.total_batch_count} "
                        f"({batch_rate:.1f}%), "
                        f"Total <sc> utterances: {self.sc_total_count}"
                    )

        # Standard training forward pass (unchanged from base trainer)
        with torch.set_grad_enabled(is_training):
            outputs = self.model(
                x=feature,
                x_lens=feature_lens,
                texts=texts,
                prune_range=self.cfg.trainer.prune_range,
                am_scale=self.cfg.trainer.am_scale,
                lm_scale=self.cfg.trainer.lm_scale,
                return_dict=True,
            )
            simple_loss = outputs["simple_loss"]
            pruned_loss = outputs["pruned_loss"]
            ctc_loss = outputs["ctc_loss"]

            loss = 0.0

            if simple_loss:
                s = self.cfg.trainer.simple_loss_scale
                # take down the scale on the simple loss from 1.0 at the start
                # to simple_loss scale by warm_step.
                simple_loss_scale = (
                    s
                    if batch_idx_train >= warm_step
                    else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
                )
                pruned_loss_scale = (
                    1.0
                    if batch_idx_train >= warm_step
                    else 0.1 + 0.9 * (batch_idx_train / warm_step)
                )
                loss += (
                    simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
                )

            if ctc_loss:
                loss += self.cfg.trainer.ctc_loss_scale * ctc_loss

        assert loss.requires_grad == is_training

        # Track metrics
        info = MetricsTracker()
        num_frames = (feature_lens // 4).sum().item()
        num_samples = batch_size
        info.set_value("frames", num_frames, normalization="sum")
        info.set_value("samples", num_samples, normalization="sum")

        # Note: We use reduction=sum while computing the loss.
        info.set_value(
            "loss", loss.detach().cpu().item() / num_frames, normalization="frame_avg"
        )
        if simple_loss:
            info.set_value(
                "simple_loss",
                simple_loss.detach().cpu().item() / num_frames,
                normalization="frame_avg",
            )
            info.set_value(
                "pruned_loss",
                pruned_loss.detach().cpu().item() / num_frames,
                normalization="frame_avg",
            )
        if ctc_loss:
            info.set_value(
                "ctc_loss",
                ctc_loss.detach().cpu().item() / num_frames,
                normalization="frame_avg",
            )

        return loss, info

    def validate(self, epoch: int):
        """
        Validation is provided by BaseTrainer.

        Override in a subclass if you need SOT-specific validation logic
        (e.g., computing speaker change detection metrics).
        """
        return super().validate(epoch)
    
    def run(self):
        """Override run to print SOT statistics at the end."""
        super().run()
        
        # Print final SOT statistics
        if self.rank == 0 and self.total_batch_count > 0:
            logging.info("=" * 80)
            logging.info("Final SOT Training Statistics:")
            logging.info(f"  Total batches processed: {self.total_batch_count}")
            logging.info(f"  Batches with <sc> token: {self.sc_batch_count}")
            logging.info(f"  Total utterances with <sc>: {self.sc_total_count}")
            batch_rate = self.sc_batch_count / self.total_batch_count * 100
            logging.info(f"  Batch coverage: {batch_rate:.2f}%")
            logging.info("=" * 80)

