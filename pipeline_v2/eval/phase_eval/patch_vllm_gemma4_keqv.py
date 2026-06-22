#!/usr/bin/env python3
"""Patch vLLM Gemma4 k_eq_v attention split for local evaluation.

vLLM 0.21.0's Gemma4 implementation can emit Q+K for Gemma4 layers with
attention_k_eq_v=true, while the forward path still splits Q+K+V. This local
patch makes those layers split Q+K and reuse K as V.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


ORIGINAL = """        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
"""

PATCHED = """        qkv, _ = self.qkv_proj(hidden_states)
        if self.use_k_eq_v and qkv.shape[-1] == self.q_size + self.kv_size:
            q, k = qkv.split([self.q_size, self.kv_size], dim=-1)
            v = k
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
"""


def find_vllm_gemma4_path() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vllm is not importable in this Python environment")
    package_root = Path(spec.origin).resolve().parent
    path = package_root / "model_executor" / "models" / "gemma4.py"
    if not path.exists():
        raise RuntimeError(f"vLLM Gemma4 source not found: {path}")
    return path


def patch_file(path: Path, *, dry_run: bool) -> str:
    text = path.read_text(encoding="utf-8")
    if PATCHED in text:
        return "already_patched"
    if ORIGINAL not in text:
        raise RuntimeError(f"target snippet not found in {path}")
    backup = path.with_suffix(path.suffix + ".codex_backup")
    if not dry_run:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    return f"patched backup={backup}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch local vLLM Gemma4 k_eq_v split.")
    parser.add_argument("--path", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = Path(args.path) if args.path else find_vllm_gemma4_path()
    print(patch_file(path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
