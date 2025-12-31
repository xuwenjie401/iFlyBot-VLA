import torch

VLM_OLD_TO_NEW = {
    "proprios.weight": "proprios_embed.weight",
    "proprios.bias": "proprios_embed.bias",
}

DIT_OLD_TO_NEW = {
    "action_encoder.weight": "action_embed.weight",
    "action_encoder.bias": "action_embed.bias",
    "action_decoder.weight": "action_out_proj.weight",
    "action_decoder.bias": "action_out_proj.bias",
    "proprio_encoder.weight": "proprio_embed.weight",
    "proprio_encoder.bias": "proprio_embed.bias",
    "time_encoder.0.weight": "time_mlp.0.weight",
    "time_encoder.0.bias": "time_mlp.0.bias",
    "time_encoder.2.weight": "time_mlp.2.weight",
    "time_encoder.2.bias": "time_mlp.2.bias",
}


def rule_based_rewrite_key(old_key: str) -> str:
    parts = old_key.split(".")

    if len(parts) < 4:
        return DIT_OLD_TO_NEW.get(old_key, old_key)

    if len(parts) >= 4 and parts[0] == "net":
        layer = parts[1]
        func = parts[2]
        rest = ".".join(parts[3:])
        return f"blocks.{layer}.{func}.{rest}"

    # 其它 key 原样返回
    return old_key


def remap_ckpt_keys(old_model):
    new_vlm = {}
    for k, v in old_model["vlm"].items():
        # 如果在映射表里，用新 key；否则保持原 key
        new_k = VLM_OLD_TO_NEW.get(k, k)
        new_vlm[new_k] = v

    new_dit = {}
    for k, v in old_model["action_head"].items():
        new_dit[rule_based_rewrite_key(k)] = v

    return {
        "vlm": new_vlm,
        "action_head": new_dit
    }


def model_to_tree_json(obj):
    """
    通用递归转换：
    - Tensor -> {"shape": "dim0xdim1x..."}
    - dict   -> 递归 dict
    """
    if torch.is_tensor(obj):
        # 人眼友好的单行表示
        shape_str = "x".join(str(d) for d in obj.shape)
        return {"shape": shape_str}

    if isinstance(obj, dict):
        return {k: model_to_tree_json(v) for k, v in obj.items()}

    raise TypeError(
        f"Unsupported type {type(obj)}; expected only dict or torch.Tensor"
    )


if __name__ == "__main__":
    import json
    ckpt_path = "/wx-mix01/sppro/permanent/wjxu22/model/iflybot_vla_ckpts/1104_rel_eef_quat_loss0d4614_step39999.pt"
    json_path = "/wx-mix01/sppro/permanent/wjxu22/codes/iFlyBotVLA/debug/transfered_ckpt_dict.json"

    state_dict = torch.load(ckpt_path, map_location="cpu")
    trans_model_dict = remap_ckpt_keys(state_dict["model"])
    tree = model_to_tree_json(trans_model_dict)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    print(f"[OK] tree json saved: {json_path}")


