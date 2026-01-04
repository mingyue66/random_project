from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as dist_all_gather


def all_gather_embeddings(emb: torch.Tensor) -> torch.Tensor:
    """
    Differentiable all-gather that tolerates different local batch sizes.
    Each rank ends up with a (global_B, D) tensor that
    participates fully in back-prop.
    """
    if not dist.is_initialized():
        return emb

    # 1) Share local batch sizes
    local_B = torch.tensor([emb.size(0)], device=emb.device, dtype=torch.long)
    B_list = [torch.zeros_like(local_B) for _ in range(dist.get_world_size())]
    dist.all_gather(B_list, local_B)
    B_list = [b.item() for b in B_list]
    max_B = max(B_list)

    # 2) Pad along dim 0 to max_B *explicitly*
    if local_B < max_B:
        pad = emb.new_zeros((max_B - local_B, emb.size(1)))  # [pad_rows, D]
        emb = torch.cat([emb, pad], dim=0)  # shape = (max_B, D)

    # 3) Differentiable all-gather (same shape on every rank)
    gathered = dist_all_gather(emb)  # [world, max_B, D]

    # 4) Remove the padding for each rank, then concat
    chunks = [g[:B] for g, B in zip(gathered, B_list)]  # exact per-rank slice
    return torch.cat(chunks, dim=0)  # (global_B, D)


def ddp_all_gather_variable_tensor_to_rank0(
    tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Gather tensors of potentially different first-dim sizes across all ranks to rank-0.
    return a tensor of shape (global_B, D) on rank 0, and None on other ranks
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = tensor.device

    # Step 1: Gather sizes
    local_size = torch.tensor([tensor.size(0)], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    sizes = [int(sz.item()) for sz in all_sizes]
    max_size = max(sizes)

    # Step 2: Pad local tensor
    pad_len = max_size - tensor.size(0)
    if pad_len > 0:
        padding = torch.zeros(
            (pad_len, *tensor.shape[1:]), dtype=tensor.dtype, device=device
        )
        tensor = torch.cat([tensor, padding], dim=0)

    # Step 3: Gather padded tensors
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)

    # Step 4: Truncate to actual sizes on rank 0
    if rank == 0:
        out = [g[:s] for g, s in zip(gather_list, sizes)]
        out = torch.cat(out, dim=0)
        return out
    else:
        return None
