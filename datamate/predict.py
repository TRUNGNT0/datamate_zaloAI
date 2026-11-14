"""CLI entry point for Datamate ZAIC Hybrid Pipeline.

Supports running SQL prediction over shops with DuckDB, LLM-based rewrite, multi-sampling,
automatic error repair, consensus voting, abstention, and canonical views/debt layers.

Usage:
    python -m datamate.predict --data dataset/public_test --output runs/public --model Qwen/Qwen3.5-9B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datamate.config import Config
from datamate.llm import create_backend
from datamate.agent.pipeline import run_inference


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Datamate ZAIC 2026 Inference")
    p.add_argument("--data", type=Path, required=True, help="Path to data split folder (e.g. dataset/public_test)")
    p.add_argument("--output", type=Path, required=True, help="Output directory for submission.jsonl and logs")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B", help="Model name or local path")
    p.add_argument("--lora", type=str, default=None, help="Optional LoRA adapter directory")
    p.add_argument("--backend", choices=("local", "gemini"), default="local", help="LLM backend")
    p.add_argument("--local-engine", choices=("auto", "vllm", "hf"), default="auto", help="Local inference engine")
    p.add_argument("--hf-quant", choices=("auto", "4bit", "none"), default="auto", help="HF quantization mode")
    p.add_argument("--k", "--n-samples", dest="n_samples", type=int, default=3, help="Number of SQL samples per question")
    p.add_argument("--agree-min", type=int, default=2, help="Minimum matching consensus needed to answer")
    p.add_argument("--max-retry", type=int, default=2, help="Max self-repair turns for failed SQL queries")
    p.add_argument("--views", choices=("cached", "llm", "none"), default="cached", help="Semantic views mode")
    p.add_argument("--views-from", type=Path, default=Path("configs/public_v9"), help="Directory containing saved views JSONs")
    p.add_argument("--fewshot", type=Path, default=Path("assets/fewshot_bank.jsonl"), help="Few-shot exemplars bank file")
    p.add_argument("--limit", type=int, default=None, help="Max questions to run")
    p.add_argument("--max-biz", type=int, default=None, help="Max businesses to run")
    p.add_argument("--time-budget", type=float, default=None, help="Max execution time in seconds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--shard", default="0/1", help="i/n shard for distributed GPU processes")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(parents=True, exist_ok=True)

    cfg = Config(
        test_dir=str(args.data),
        train_dir=str(args.data.parent / "train") if (args.data.parent / "train").exists() else str(args.data),
        out_dir=str(args.output),
        database_dir=str(args.output / "databases"),
        manifest_dir=str(args.output / "manifests") if (args.output / "manifests").exists() else None,
        backend=args.backend,
        query_engine="sql",
        local_model=args.model,
        local_lora=args.lora,
        local_engine=args.local_engine,
        hf_quant=args.hf_quant,
        n_samples=args.n_samples,
        agree_min=args.agree_min,
        max_retry=args.max_retry,
        temperature=args.temperature,
        limit_test=args.limit,
        max_biz=args.max_biz,
        time_budget_s=args.time_budget,
        views=args.views,
        views_from=str(args.views_from) if args.views_from and args.views_from.is_dir() else None,
        fewshot_bank=str(args.fewshot) if args.fewshot and args.fewshot.is_file() else None,
    )

    print(f"=== Datamate ZAIC 2026 Pipeline ===")
    print(f"Data: {args.data} | Output: {args.output}")
    print(f"Model: {args.model} (engine={args.local_engine}, quant={args.hf_quant})")
    print(f"Sampling: {args.n_samples} samples, agree_min={args.agree_min}, views={args.views}")

    backend = create_backend(cfg) if cfg.backend else None
    subs, logs, problems = run_inference(cfg, backend)

    elapsed = round(time.time() - t0, 1)
    print(f"Hoàn tất sau {elapsed}s | Số câu: {len(subs)} | Vấn đề kiểm định: {len(problems)}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
