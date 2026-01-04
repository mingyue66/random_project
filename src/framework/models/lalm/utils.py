import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def forward(
    self,
    input_features,
    attention_mask=None,
    head_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    r"""
    Args:
        input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
            Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
            obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a
            `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec libary (`pip install torchcodec`) or
            the soundfile library (`pip install soundfile`). To prepare the array into
            `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        attention_mask (`torch.Tensor`)`, *optional*):
            Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
            but it is not used. By default the silence in the input log mel spectrogram are ignored.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """

    # expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
    # if input_features.shape[-1] != expected_seq_length:
    #     raise ValueError(
    #         f"Whisper expects the mel input features to be of length {expected_seq_length}, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
    #     )

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    inputs_embeds = nn.functional.gelu(self.conv1(input_features))
    inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    all_positions = torch.arange(
        inputs_embeds.shape[1], device=inputs_embeds.device
    )  # truncated to the exact length

    hidden_states = inputs_embeds + self.embed_positions(all_positions)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )

    encoder_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
        to_drop = False
        if self.training:
            dropout_probability = torch.rand([])
            if dropout_probability < self.layerdrop:  # skip the layer
                to_drop = True

        if to_drop:
            layer_outputs = (None, None)
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    hidden_states = self.layer_norm(hidden_states)
    if output_hidden_states:
        encoder_states = encoder_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, encoder_states, all_attentions] if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=encoder_states,
        attentions=all_attentions,
    )


def replace_whisper_encoder_forward():
    """
    This function monkey patches the forward method of the whisper encoder.
    To be called before the model is loaded, it changes whisper to process audio with any length < 30s.
    """
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    WhisperEncoder.forward = forward


def preprocess_text_and_audio_impl(
    *,
    messages: List[List[Dict[str, str]]],
    tokenizer,
    llm,
    audio_token: str,
    audio_features: Optional[torch.Tensor],  # [B, T, H] after projector
    audio_feature_lens: Optional[torch.Tensor],  # [B]
    max_length: int = 256,
    tag_audio_boundary: bool = False,
    is_training: bool = False,
    audio_tag_embedding: Optional[torch.Tensor] = None,  # [2, H]
    chat_template: Optional[str] = None,
):
    """
    Build input_ids, input_embeds, attention_mask (2D), and labels for the LLM.

    - Expands a single audio token in each message to multiple tokens based on
      audio_feature_lens and replaces those token embeddings with audio_features.
    - Supports optional insertion of start/end audio boundary embeddings.
    - Uses only a 2D attention mask (no custom 4D masking).
    """

    batch_size = len(messages)

    # Resolve ids and basic settings
    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    if audio_token_id is None or audio_token_id < 0:
        raise ValueError(
            f"audio_token '{audio_token}' is not in the tokenizer vocabulary."
        )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id must be set.")

    # Training vs inference switches
    add_generation_prompt = not is_training
    prepare_label = is_training

    # Prepare audio features
    if audio_features is not None and audio_feature_lens is not None:
        if audio_features.dim() != 3:
            raise ValueError(
                f"audio_features must be [B, T, H], got {tuple(audio_features.shape)}"
            )
        if (
            audio_features.size(0) != batch_size
            or audio_feature_lens.size(0) != batch_size
        ):
            raise ValueError(
                "Batch size mismatch between messages and audio features/lens."
            )
        device = audio_features.device
        max_len = int(audio_feature_lens.max().item())
        audio_features = audio_features[:, :max_len]
        if tag_audio_boundary and audio_tag_embedding is not None:
            # [2, H] -> [2, 1, H], then concat
            audio_tag_embedding = audio_tag_embedding.unsqueeze(1).to(
                audio_features.dtype
            )
            audio_features = torch.cat(
                [
                    audio_tag_embedding[[0]].repeat(batch_size, 1, 1),
                    audio_features,
                    audio_tag_embedding[[1]].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )
            audio_feature_lens = audio_feature_lens + 2
    else:
        device = torch.device("cpu")
        audio_feature_lens = torch.zeros((batch_size,), dtype=torch.long, device=device)
        audio_features = None

    # Build per-sample ids by expanding a single audio token to L occurrences
    lens_list = audio_feature_lens.tolist()
    input_ids_list: List[List[int]] = []
    max_input_len = 0
    for i, msg in enumerate(messages):
        try:
            s = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                padding="do_not_pad",
                truncation=False,
                max_length=max_length,
            )
        except Exception:
            # Fallback: join roles/contents simply
            s = "".join([f"{turn['role']}\n{turn['content']}\n" for turn in msg])
        ids = tokenizer.encode(s, add_special_tokens=False)
        try:
            k = ids.index(audio_token_id)
            L = int(lens_list[i])
            out = ids[:k] + [audio_token_id] * L + ids[k + 1 :]
        except ValueError:
            out = ids
        if max_length is not None:
            out = out[:max_length]
        input_ids_list.append(out)
        if len(out) > max_input_len:
            max_input_len = len(out)

    # Pad ids
    if tokenizer.padding_side == "right":
        padded = [ids + [pad_id] * (max_input_len - len(ids)) for ids in input_ids_list]
    else:
        padded = [[pad_id] * (max_input_len - len(ids)) + ids for ids in input_ids_list]
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    padding_mask = input_ids == pad_id

    # Text embeddings
    input_embeds = llm.get_input_embeddings()(input_ids)

    # Replace audio token embeddings per sample to avoid scatter-count mismatches
    if audio_features is not None:
        input_embeds = input_embeds.clone()  # avoid in-place on view
        audio_features = audio_features.to(input_embeds.dtype)
        hidden_size = input_embeds.size(-1)
        if audio_features.size(-1) != hidden_size:
            raise ValueError(
                f"Audio feature dim mismatch: {audio_features.size(-1)} != LLM hidden {hidden_size}"
            )
        for i in range(batch_size):
            pos = (input_ids[i] == audio_token_id).nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                continue
            L = int(audio_feature_lens[i].item())
            K = min(L, pos.numel())
            if K > 0:
                input_embeds[i, pos[:K]] = audio_features[i, :K]
            if K < pos.numel():
                # Leave remaining positions as their text-token embeddings
                logging.debug(
                    f"Sample {i}: only filled {K}/{pos.numel()} audio tokens from {L} frames"
                )

    # Labels and 2D attention mask
    if prepare_label:
        labels = input_ids.clone().to(torch.long)
        labels[padding_mask] = IGNORE_TOKEN_ID
        # Best-effort: ignore prompt up to assistant start marker
        try:
            assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if (
                assistant_token_id is not None
                and assistant_token_id != tokenizer.unk_token_id
                and im_start_id is not None
            ):
                rows, cols = torch.where(input_ids == assistant_token_id)
                for r, c in zip(rows.tolist(), cols.tolist()):
                    if c > 1 and input_ids[r, c - 1].item() == im_start_id:
                        labels[r, : c + 2] = IGNORE_TOKEN_ID
        except Exception:
            pass
    else:
        labels = None

    attention_mask = (~padding_mask).to(input_embeds.dtype)
    position_ids = None  # will be derived from input_embeds
    return input_ids, input_embeds, attention_mask, labels, position_ids


def preprocess_text_and_audio_packed_sdpa(
    *,
    messages: List[List[Dict[str, str]]],
    tokenizer,
    llm,
    audio_token: str,
    audio_features: Optional[torch.Tensor],  # [B, T, H] after projector
    audio_feature_lens: Optional[torch.Tensor],  # [B]
    max_length: Optional[int] = None,
    is_training: bool = False,
    chat_template: Optional[str] = None,
    max_total_length: Optional[int] = None,
):
    """
    Sequence packing variant: concatenate all samples in the batch into a single
    long sequence without padding. Prevents per-sample padding overhead.

    Note: This does not add a block-diagonal attention mask; as in standard LM
    packing, tokens can attend across sample boundaries. If strict isolation is
    required, a 4D block mask would be needed and supported by specific models.
    """

    batch_size = len(messages)
    if max_total_length is not None and max_total_length <= 0:
        raise ValueError("max_total_length must be a positive integer when provided.")

    audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
    if audio_token_id is None or audio_token_id < 0:
        raise ValueError(
            f"audio_token '{audio_token}' is not in the tokenizer vocabulary."
        )

    add_generation_prompt = not is_training

    # Prepare audio features and lengths (assume provided)
    device = audio_features.device

    # Build chat strings via template, then batch tokenize (no padding requested)
    texts: List[str] = []
    for msg in messages:
        try:
            s = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                padding="do_not_pad",
                truncation=False,
                max_length=max_length,
            )
        except Exception:
            s = "".join([f"{turn['role']}\n{turn['content']}\n" for turn in msg])
        texts.append(s)
    tok = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_tensors=None,
    )
    ids_lists = tok.input_ids

    # Compute per-sample text embeddings without padding by flattening then slicing
    lens = [len(x) for x in ids_lists]
    flat_ids = torch.tensor(
        [t for seq in ids_lists for t in seq], dtype=torch.long, device=device
    )
    if flat_ids.numel() > 0:
        flat_embeds = llm.get_input_embeddings()(flat_ids)  # [Ltot, H]
    else:
        H = llm.get_input_embeddings().weight.size(1)
        flat_embeds = torch.empty(
            (0, H), device=device, dtype=llm.get_input_embeddings().weight.dtype
        )
    # Build per-sample views
    text_embeds_full = []
    offset = 0
    for ln in lens:
        text_embeds_full.append(flat_embeds[offset : offset + ln])
        offset += ln
    out_embeds: List[torch.Tensor] = []
    out_ids: List[torch.Tensor] = []
    total_tokens = 0

    for i in range(batch_size):
        cur_ids = torch.tensor(ids_lists[i], dtype=torch.long, device=device)
        # find first audio token position
        audio_pos = (cur_ids == audio_token_id).nonzero(as_tuple=False).squeeze(-1)
        if audio_pos.numel() > 0:
            k = int(audio_pos[0].item())
            left_ids = cur_ids[:k]
            right_ids = cur_ids[k + 1 :]
            emb_i = text_embeds_full[i]
            left_emb = emb_i[:k]
            right_emb = emb_i[k + 1 : len(cur_ids)]
        else:
            left_ids = cur_ids
            right_ids = cur_ids.new_empty((0,))
            emb_i = text_embeds_full[i]
            left_emb = emb_i[: len(cur_ids)]
            right_emb = emb_i[: len(cur_ids)][0:0]

        L_audio = int(audio_feature_lens[i].item())
        audio_emb = audio_features[i, :L_audio]
        mid_ids = torch.full(
            (L_audio,), audio_token_id, dtype=left_ids.dtype, device=left_ids.device
        )

        out_ids_i = torch.cat([left_ids, mid_ids, right_ids], dim=0)
        embeds = torch.cat([left_emb, audio_emb, right_emb], dim=0)

        out_ids.append(out_ids_i)
        out_embeds.append(embeds)

        total_tokens += out_ids_i.size(0)
        if max_total_length is not None and total_tokens > max_total_length:
            overflow = total_tokens - max_total_length
            logging.warning(
                "Packed token count %d exceeds max_total_length=%d; truncating last "
                "segment by %d tokens to stay within limits.",
                total_tokens,
                max_total_length,
                overflow,
            )
            keep = out_ids_i.size(0) - overflow
            if keep <= 0:
                out_ids.pop()
                out_embeds.pop()
            else:
                out_ids[-1] = out_ids[-1][:keep]
                out_embeds[-1] = out_embeds[-1][:keep]
            total_tokens = max_total_length
            break

    # Concatenate across samples (packed)
    input_embeds = torch.cat(out_embeds, dim=0).unsqueeze(0)  # [1, L, H]
    input_ids = torch.cat(out_ids, dim=0).unsqueeze(0)  # [1, L]
    # Always build block-diagonal causal mask for isolation across packed samples.
    L = input_ids.size(1)
    # start with all disallowed; keep dtype consistent with model embeddings
    mask = torch.full(
        (L, L),
        float("-inf"),
        device=device,
        dtype=input_embeds.dtype,
    )
    # mark segment boundaries
    seg_lens = [t.size(0) for t in out_embeds]
    starts = []
    s = 0
    for ln in seg_lens:
        starts.append(s)
        s += ln
    # within each block, allow only causal (lower-triangular including diag)
    for st, ln in zip(starts, seg_lens):
        ed = st + ln
        block = mask[st:ed, st:ed]
        lower = torch.tril(torch.ones((ln, ln), device=device), diagonal=0).bool()
        block[lower] = 0.0
    attention_mask = mask.unsqueeze(0).unsqueeze(
        0
    )  # [1, 1, L, L], additive mask (0 allow, -inf disallow)
    # Labels consistent with preprocess_text_and_audio_impl, but applied per segment
    if is_training:
        labels = input_ids.clone().to(torch.long)
        try:
            assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if (
                assistant_token_id is not None
                and assistant_token_id != tokenizer.unk_token_id
                and im_start_id is not None
            ):
                # compute segment starts/lens if not present
                seg_lens = [t.size(0) for t in out_embeds]
                starts = []
                s = 0
                for ln in seg_lens:
                    starts.append(s)
                    s += ln
                row = input_ids[0]
                for st, ln in zip(starts, seg_lens):
                    ed = st + ln
                    seg = row[st:ed]
                    cols = (
                        (seg == assistant_token_id).nonzero(as_tuple=False).squeeze(-1)
                    )
                    for c_rel in cols.tolist():
                        if c_rel > 0 and seg[c_rel - 1].item() == im_start_id:
                            c_abs = st + c_rel
                            labels[0, st : c_abs + 2] = IGNORE_TOKEN_ID
        except Exception:
            pass
    else:
        labels = None

    # Build per-token position_ids resetting at each segment boundary
    seg_lens = [t.size(0) for t in out_embeds]
    pos_list = [torch.arange(ln, device=device, dtype=torch.long) for ln in seg_lens]
    position_ids = (
        torch.cat(pos_list, dim=0).unsqueeze(0) if len(pos_list) > 0 else None
    )
    return input_ids, input_embeds, attention_mask, labels, position_ids
