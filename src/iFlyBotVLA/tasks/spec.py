from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 10
    do_sample: bool = False
    temperature: float = 0.7
    eos_token_ids: Optional[Sequence[int]] = None

    def to_kwargs(self) -> dict:
        kw = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            return_dict_in_generate=True,
        )
        if self.eos_token_ids is not None:
            kw["eos_token_id"] = list(self.eos_token_ids)
        return kw


@dataclass(frozen=True)
class TaskSpec:
    """一个任务 = decoding config + KV selection策略（以及未来可扩展的解析规则等）"""
    name: str
    gen_cfg: GenerationConfig
    kv_selector: "KVSelector"  # forward ref
