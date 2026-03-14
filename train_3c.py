import argparse
import os
import subprocess
import sys
import tempfile
from typing import Dict, Any, Tuple, Optional

import torch


def unwrap_state_dict(ckpt: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], str]:
    if not isinstance(ckpt, dict):
        raise ValueError("Invalid checkpoint format.")
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"], "model_state_dict"
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"], "state_dict"
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"], "model"
    return ckpt, ""


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def find_head_keys(
    src_state: Dict[str, torch.Tensor],
    head_weight_key: str = "",
    head_bias_key: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    if head_weight_key:
        if head_weight_key not in src_state:
            raise KeyError(f"head_weight_key not found: {head_weight_key}")
        b = head_bias_key if head_bias_key in src_state else None
        return head_weight_key, b

    # 自动找 [2, C, ...] 的最终分类头
    cands = []
    for k, v in src_state.items():
        if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] == 2:
            score = sum(x in k.lower() for x in ["seg", "head", "out", "final", "classifier"])
            cands.append((score, k))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0], reverse=True)
    w = cands[0][1]
    b = w.replace(".weight", ".bias")
    if b in src_state and torch.is_tensor(src_state[b]) and src_state[b].ndim == 1 and src_state[b].shape[0] == 2:
        return w, b
    return w, None


def remap_2c_to_3c(src_ckpt: str, dst_ckpt: str, head_weight_key: str = "", head_bias_key: str = ""):
    raw = torch.load(src_ckpt, map_location="cpu")
    src_state, container_key = unwrap_state_dict(raw)
    src_state = strip_module_prefix(src_state)

    w_key, b_key = find_head_keys(src_state, head_weight_key, head_bias_key)
    if w_key is None:
        raise RuntimeError("Cannot find 2-class head. Please pass --head_weight_key explicitly.")

    w = src_state[w_key]
    if w.shape[0] != 2:
        raise RuntimeError(f"Head first dim is not 2: {w_key}, shape={tuple(w.shape)}")

    new_w = torch.zeros((3, *w.shape[1:]), dtype=w.dtype)
    new_w[0] = w[0]  # bg
    new_w[1] = w[1]  # LAD <- vessel
    new_w[2] = w[1]  # LCX <- vessel
    src_state[w_key] = new_w

    if b_key:
        b = src_state[b_key]
        new_b = torch.zeros((3,), dtype=b.dtype)
        new_b[0] = b[0]
        new_b[1] = b[1]
        new_b[2] = b[1]
        src_state[b_key] = new_b

    # 写回原容器结构，尽量兼容原加载逻辑
    if container_key:
        raw[container_key] = src_state
        out_obj = raw
    else:
        out_obj = src_state

    os.makedirs(os.path.dirname(dst_ckpt), exist_ok=True) if os.path.dirname(dst_ckpt) else None
    torch.save(out_obj, dst_ckpt)

    print(f"[OK] converted: {dst_ckpt}")
    print(f"[OK] head weight: {w_key}")
    if b_key:
        print(f"[OK] head bias: {b_key}")


def main():
    parser = argparse.ArgumentParser("2c->3c launcher")
    parser.add_argument("--entry", type=str, default="train.py", help="training entry script")
    parser.add_argument("--src_ckpt_2c", type=str, required=True)
    parser.add_argument("--dst_ckpt_3c", type=str, default="")
    parser.add_argument("--head_weight_key", type=str, default="")
    parser.add_argument("--head_bias_key", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    args, passthrough = parser.parse_known_args()

    if not args.dst_ckpt_3c:
        src_dir = os.path.dirname(os.path.abspath(args.src_ckpt_2c))
        args.dst_ckpt_3c = os.path.join(src_dir, "init_3c_from_2c.pth")

    remap_2c_to_3c(
        src_ckpt=args.src_ckpt_2c,
        dst_ckpt=args.dst_ckpt_3c,
        head_weight_key=args.head_weight_key,
        head_bias_key=args.head_bias_key,
    )

    # 适配你当前 train.py: --pretrain(开关) + --load(路径)
    cmd = [sys.executable, args.entry, "--pretrain", "--load", args.dst_ckpt_3c]
    cmd.extend(passthrough)

    print("[CMD]", " ".join(cmd))
    if args.dry_run:
        return

    ret = subprocess.run(cmd, check=False)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()