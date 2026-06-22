#!/usr/bin/env python3
"""Wait for a pipeline adapter, run phase eval, and optionally upload it.

This is intentionally separate from the training runner so an already-running
recipe can continue untouched while the finalization step waits in the
background.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
DEFAULT_PYTHON = sys.executable
DEFAULT_MODEL = "/workspace/modelscope_cache/unsloth/gemma-4-31B-it"


def print_json(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + " " + json.dumps(payload, ensure_ascii=False), flush=True)


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print_json("[postprocess/run]", {"command": command, "log": str(log_path)})
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[postprocess/start] " + json.dumps({"time": start, "command": command}, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        returncode = proc.wait()
        elapsed = time.time() - start
        log.write(
            "\n[postprocess/end] "
            + json.dumps({"returncode": returncode, "elapsed_sec": elapsed}, ensure_ascii=False)
            + "\n"
        )
    result = {"command": command, "returncode": returncode, "elapsed_sec": round(elapsed, 3), "log": str(log_path)}
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return result


def adapter_ready(path: Path) -> bool:
    if not path.exists():
        return False
    required_any = ["adapter_model.safetensors", "adapter_model.bin"]
    return (path / "adapter_config.json").exists() and any((path / name).exists() for name in required_any)


def wait_for_adapter(path: Path, *, timeout_sec: int, poll_sec: int, stable_sec: int) -> None:
    start = time.time()
    last_mtime = 0.0
    stable_since: float | None = None
    while True:
        if adapter_ready(path):
            mtimes = [p.stat().st_mtime for p in path.glob("*") if p.is_file()]
            current_mtime = max(mtimes) if mtimes else path.stat().st_mtime
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                stable_since = time.time()
            elif stable_since is not None and time.time() - stable_since >= stable_sec:
                print_json("[postprocess/adapter_ready]", {"path": str(path), "stable_sec": stable_sec})
                return
        elapsed = time.time() - start
        if timeout_sec > 0 and elapsed > timeout_sec:
            raise TimeoutError(f"Timed out waiting for adapter: {path}")
        print_json("[postprocess/wait]", {"path": str(path), "elapsed_sec": round(elapsed, 1)})
        time.sleep(poll_sec)


def write_upload_readme(path: Path, *, model_id: str, base_model: str, eval_dir: Path, run_root: Path) -> None:
    lines = [
        "---",
        f"base_model: {base_model}",
        "library_name: peft",
        "tags:",
        "- gemma4",
        "- lora",
        "- qlora",
        "- korean-webnovel",
        "- unsloth",
        "---",
        "",
        f"# {model_id}",
        "",
        "Gemma 4 31B Korean webnovel style LoRA adapter trained with the pipeline_v2 full recipe.",
        "",
        "## Recipe",
        "",
        "- Stage 1: CPT/SFT-lite with row-type balanced sampling and anti-slop unlikelihood.",
        "- Stage 2: Format and continuation SFT.",
        "- Stage 3: DPO preference training on rewrite and bad-style pairs.",
        "- Stage 4: Generate GRPO with GUI-style distribution rewards.",
        "- Stage 5: Rewrite GRPO with GUI-style, edit-amount, and improvement rewards.",
        "",
        "## Local Artifacts",
        "",
        f"- run root: `{run_root}`",
        f"- phase eval: `{eval_dir}`",
    ]
    (path / "README.md").write_text("\n".join(lines), encoding="utf-8")


def upload_to_modelscope(
    *,
    adapter_path: Path,
    model_id: str,
    token: str,
    endpoint: str,
    base_model: str,
    eval_dir: Path,
    run_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    write_upload_readme(adapter_path, model_id=model_id, base_model=base_model, eval_dir=eval_dir, run_root=run_root)
    if dry_run:
        return {"dry_run": True, "model_id": model_id, "adapter_path": str(adapter_path)}
    from modelscope.hub.api import HubApi

    api = HubApi(endpoint=endpoint)
    token_arg = token or None
    if token:
        api.login(access_token=token, endpoint=endpoint)
    try:
        api.create_model(model_id=model_id, visibility=5, license="Apache License 2.0", token=token_arg, endpoint=endpoint)
        created = True
    except Exception as exc:
        created = False
        print_json("[postprocess/modelscope_create_skip]", {"model_id": model_id, "error": type(exc).__name__})
    result = api.upload_folder(
        repo_id=model_id,
        folder_path=str(adapter_path),
        path_in_repo="",
        repo_type="model",
        token=token_arg,
        commit_message="Upload Gemma4 webnovel LoRA adapter",
        ignore_patterns=["checkpoint-*", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"],
        max_workers=8,
    )
    return {"model_id": model_id, "created": created, "result": str(result)}


def cleanup_checkpoints(run_root: Path, *, keep_latest: int) -> list[str]:
    removed: list[str] = []
    if keep_latest < 0:
        return removed
    for stage_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        checkpoints = sorted(
            [p for p in stage_dir.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for checkpoint in checkpoints[keep_latest:]:
            shutil.rmtree(checkpoint, ignore_errors=True)
            removed.append(str(checkpoint))
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for final adapter, run eval, and upload to ModelScope.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default="full_lora_recipe_v1")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--wait-timeout-sec", type=int, default=0)
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--stable-sec", type=int, default=120)
    parser.add_argument("--eval-count", type=int, default=30)
    parser.add_argument("--eval-dir", default="")
    parser.add_argument("--skip-eval-prepare", action="store_true")
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--generation-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--max-new-tokens-rewrite", type=int, default=4096)
    parser.add_argument("--max-new-tokens-generate", type=int, default=4096)
    parser.add_argument("--modelscope-model-id", default=os.environ.get("MODELSCOPE_MODEL_ID", ""))
    parser.add_argument("--modelscope-endpoint", default=os.environ.get("MODELSCOPE_ENDPOINT", "https://modelscope.ai"))
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--cleanup-checkpoints-keep-latest", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "HF_HOME": os.environ.get("HF_HOME", "/workspace/.hf_home"),
        "UNSLOTH_DISABLE_STATISTICS": "1",
        "UNSLOTH_USE_MODELSCOPE": "1",
        "TORCHDYNAMO_DISABLE": "1",
    }

    wait_for_adapter(adapter_path, timeout_sec=args.wait_timeout_sec, poll_sec=args.poll_sec, stable_sec=args.stable_sec)

    commands: list[dict[str, Any]] = []
    eval_output_dir = output_dir / "phase_eval_final"
    if not args.skip_eval:
        command = [
            args.python,
            str(ROOT / "pipeline_v2/eval/phase_eval/run_phase_eval_bundle.py"),
            "--output-dir",
            str(eval_output_dir),
            "--model",
            args.model,
            "--adapter-path",
            str(adapter_path),
            "--model-label",
            args.model_label,
            "--generation-batch-size",
            str(args.generation_batch_size),
            "--generation-backend",
            args.generation_backend,
            "--rewrite-count",
            str(args.eval_count),
            "--generate-count",
            str(args.eval_count),
            "--control-count",
            str(args.eval_count),
            "--max-new-tokens-rewrite",
            str(args.max_new_tokens_rewrite),
            "--max-new-tokens-generate",
            str(args.max_new_tokens_generate),
            "--load-in-4bit",
        ]
        if args.eval_dir:
            command.extend(["--eval-dir", args.eval_dir])
        if args.skip_eval_prepare:
            command.append("--skip-prepare")
        commands.append(
            run_command(command, cwd=ROOT, env=env, log_path=output_dir / "phase_eval_final.log")
        )

    removed = cleanup_checkpoints(run_root, keep_latest=args.cleanup_checkpoints_keep_latest)
    upload_result: dict[str, Any] | None = None
    if not args.skip_upload:
        if not args.modelscope_model_id:
            raise ValueError("--modelscope-model-id or MODELSCOPE_MODEL_ID is required unless --skip-upload is set.")
        upload_result = upload_to_modelscope(
            adapter_path=adapter_path,
            model_id=args.modelscope_model_id,
            token=env.get("MODELSCOPE_API_TOKEN", ""),
            endpoint=args.modelscope_endpoint,
            base_model=args.model,
            eval_dir=eval_output_dir,
            run_root=run_root,
            dry_run=args.dry_run,
        )

    manifest = {
        "time": time.time(),
        "adapter_path": str(adapter_path),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "eval_output_dir": str(eval_output_dir),
        "commands": commands,
        "cleanup_removed": removed,
        "upload": upload_result,
        "args": {key: value for key, value in vars(args).items() if "token" not in key.lower()},
    }
    (output_dir / "postprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_json("[postprocess/done]", manifest)


if __name__ == "__main__":
    main()
