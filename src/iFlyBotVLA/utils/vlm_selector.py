from dataclasses import dataclass
from typing import Optional, Any, Dict

@dataclass(frozen=True)
class QwenCallSpec:
    need_proprios: bool = False
    need_domain_id: bool = False

class QwenSelector:
    """
    统一封装：不在主模型里写 if model_type / kwargs 分支。
    """
    def __init__(self, vlm, call_spec: QwenCallSpec):
        self.vlm = vlm
        self.spec = call_spec

    def _base_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        kw = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            image_grid_thw=batch["image_grid_thw"],
            position_ids=batch["position_ids"],
        )
        if self.spec.need_proprios:
            kw["proprios"] = batch["proprios"]
        if self.spec.need_domain_id:
            kw["domain_id"] = batch["domain_id"]
        return kw

    def forward(self, batch: Dict[str, Any]):
        kw = self._base_kwargs(batch)
        kw["labels"] = batch["labels"]
        return self.vlm(**kw)

    def generate(self, batch: Dict[str, Any], gen_cfg):
        kw = self._base_kwargs(batch)
        kw.update(gen_cfg.to_kwargs())
        return self.vlm.generate(**kw)
