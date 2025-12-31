# selectors/kv_selectors.py
from typing import Protocol, Tuple, Optional, Sequence, Dict, Any
import torch




class KVSelector(Protocol):
    def select(
        self,
        batch: Dict[str, Any],
        vlm_output: Any,
        keys: torch.Tensor,   # [L, B, S, K]
        values: torch.Tensor, # [L, B, S, K]
        mode: str,            # "train" or "infer"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

def _apply_seq_mask(keys: torch.Tensor, values: torch.Tensor, mask_bs: torch.Tensor):
    """
    keys/values: [L, B, S, K]
    mask_bs: [B, S] (bool or 0/1)
    """
    mask = mask_bs.to(dtype=keys.dtype)
    mask = mask[None, :, :, None]  # [1,B,S,1]
    return keys * mask, values * mask

def _span_mask_from_labels(labels: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """
    labels: [B, S]
    返回 mask: [B, S]，从首次出现 token_ids 的任意一个开始，到该位置结束（含），你可按需改成更复杂 span。
    """
    # 找到任一 token_id 的位置
    hit = torch.zeros_like(labels, dtype=torch.bool)
    for tid in token_ids:
        hit |= (labels == tid)

    # 如果一个 batch 全没命中，则返回全 1（也可返回全 0，看你需求）
    if not torch.any(hit):
        return torch.ones_like(labels, dtype=torch.bool)

    # 这里做一个“前缀 span”：对每个样本，找到最右边的命中位置 r，然后 mask[:r+1]=1
    B, S = labels.shape
    idx = torch.arange(S, device=labels.device)[None, :].expand(B, S)
    # 将未命中置为 -inf，求 max 得到最右命中 index
    rightmost = torch.where(hit, idx, torch.full_like(idx, -10**9)).max(dim=1).values  # [B]
    rightmost = rightmost.clamp(min=0)
    mask = idx <= rightmost[:, None]
    return mask

class AllKVSelector:
    def select(self, batch, vlm_output, keys, values, mode: str):
        return keys, values

class SpanFromLabelsSelector:
    """
    训练用：依据 labels 里的终止 token / action-token 标志来裁剪 KV。
    你可以把 token_ids 做成 per-task 的配置。
    """
    def __init__(self, token_ids: Sequence[int], detach: bool = False):
        self.token_ids = list(token_ids)
        self.detach = detach

    def select(self, batch, vlm_output, keys, values, mode: str):
        labels = batch["labels"]  # [B,S]
        mask = _span_mask_from_labels(labels, self.token_ids)
        keys2, values2 = _apply_seq_mask(keys, values, mask)
        if self.detach:
            keys2, values2 = keys2.detach(), values2.detach()
        return keys2, values2

class KeepLastNSelector:
    """
    推理用：只保留最后 N 个 token 的 KV（有效提升推理速度，适合 action token 很多的任务）
    """
    def __init__(self, n: int, detach: bool = False):
        self.n = int(n)
        self.detach = detach

    def select(self, batch, vlm_output, keys, values, mode: str):
        # keys: [L,B,S,K]
        S = keys.shape[2]
        n = min(self.n, S)
        keys2 = keys[:, :, S - n :, :]
        values2 = values[:, :, S - n :, :]
        if self.detach:
            keys2, values2 = keys2.detach(), values2.detach()
        return keys2, values2
