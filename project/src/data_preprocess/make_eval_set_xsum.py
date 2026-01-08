#!/usr/bin/env python3
"""
从 XSum 构造一个“固定 N 条”的 SFT 评测集（jsonl：instruction/input/output）。

典型用法：
  python project/src/data_preprocess/make_eval_set_xsum.py ^
    --out_path project/data/processed/eval_10.jsonl ^
    --n 10 --seed 42 --split validation ^
    --exclude_sft_path project/data/processed/sft_data.jsonl ^
    --exclude_rl_path project/data/processed/rl_prompts.jsonl

说明：
- 评测集的 prompt 形式与训练/评测脚本一致：instruction + ("\n"+input)
- 默认会从指定 split 中抽样；建议用 validation 或 test，避免与训练集混淆
- 可选排除：SFT 训练集中出现过的 prompt（强烈建议开启）
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Set

from datasets import load_dataset


DEFAULT_INSTRUCTION = "Write a news report in English based on the following summary:"


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_prompt(instruction: str, inp: str) -> str:
    return instruction + (("\n" + inp) if inp else "")


def load_exclude_prompts_sft(path: str | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"exclude_sft_path 不存在：{path}")
    out: Set[str] = set()
    for it in iter_jsonl(path):
        inst = it.get("instruction", "") or ""
        inp = it.get("input", "") or ""
        out.add(build_prompt(inst, inp))
    return out


def load_exclude_prompts_rl(path: str | None) -> Set[str]:
    # RL prompts 是 {"prompt": "..."}；我们按文本精确匹配排除
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"exclude_rl_path 不存在：{path}")
    out: Set[str] = set()
    for it in iter_jsonl(path):
        pr = it.get("prompt")
        if isinstance(pr, str) and pr.strip():
            out.add(pr.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="从 XSum 生成固定 N 条 eval jsonl（instruction/input/output）")
    ap.add_argument("--out_path", type=str, default="project/data/processed/eval_10.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    ap.add_argument("--exclude_sft_path", type=str, default=None, help="SFT 训练数据路径（用于排除 prompt 泄漏）")
    ap.add_argument("--exclude_rl_path", type=str, default=None, help="RL prompts 路径（可选排除）")

    args = ap.parse_args()
    if args.n <= 0:
        raise ValueError("--n 必须 > 0")

    random.seed(args.seed)

    exclude_sft = load_exclude_prompts_sft(args.exclude_sft_path)
    exclude_rl = load_exclude_prompts_rl(args.exclude_rl_path)

    ds = load_dataset("xsum", trust_remote_code=True)
    split = ds[args.split]

    candidates = []
    for row in split:
        summary = (row.get("summary") or "").strip()
        document = (row.get("document") or "").strip()
        if not summary or not document:
            continue
        prompt = build_prompt(args.instruction, summary)
        if prompt in exclude_sft:
            continue
        if prompt in exclude_rl:
            # 一般不会发生（prompt 模板不同），但保留这个保险
            continue
        candidates.append(
            {
                "instruction": args.instruction,
                "input": summary,
                "output": document,
            }
        )

    if len(candidates) < args.n:
        raise ValueError(f"可用候选不足：{len(candidates)} < {args.n}（split={args.split}）")

    picked = random.sample(candidates, args.n)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in picked:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"✅ wrote: {out_path}  (n={args.n}, seed={args.seed}, split={args.split})")
    if args.exclude_sft_path:
        print(f"   excluded sft prompts from: {args.exclude_sft_path} (unique={len(exclude_sft)})")
    if args.exclude_rl_path:
        print(f"   excluded rl prompts from: {args.exclude_rl_path} (unique={len(exclude_rl)})")


if __name__ == "__main__":
    main()

