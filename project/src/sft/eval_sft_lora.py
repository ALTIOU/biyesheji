#!/usr/bin/env python3
"""
SFT LoRA 模型评测脚本（无额外依赖版）

适用你的数据格式（jsonl，每行包含 instruction/input/output）：
  prompt = instruction + ("\n" + input if input else "")
  reference = output

评测内容：
- 生成：对 eval 子集做贪心/束搜索生成
- 指标：ROUGE-1 / ROUGE-2 / ROUGE-L（F1）
- 输出：若干条样例对比 + 汇总指标

用法示例：
  python project/src/sft/eval_sft_lora.py \
    --adapter_dir project/models/sft/sft_20260105_1453 \
    --data_path project/data/processed/sft_data.jsonl \
    --eval_size 200 --max_new_tokens 256 --num_print 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_hf_token() -> str | None:
    """
    尽量复用训练脚本的 token 逻辑：
    - 优先读 HF_TOKEN / HUGGINGFACE_HUB_TOKEN
    - 再尝试 HF_TOKEN_FILE / HF_HOME/token / ~/.huggingface/token / ~/.cache/huggingface/token
    - 都没有则返回 None（公共模型通常不需要 token；私有/限流再自行提供 HF_TOKEN）
    """
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token and hf_token.strip():
        return hf_token.strip()

    token_file = os.environ.get("HF_TOKEN_FILE")
    token_candidates = [
        token_file,
        (os.path.join(os.environ.get("HF_HOME", ""), "token") if os.environ.get("HF_HOME") else None),
        os.path.expanduser("~/.huggingface/token"),
        os.path.expanduser("~/.cache/huggingface/token"),
    ]
    for p in token_candidates:
        if not p:
            continue
        try:
            tok = Path(p).read_text(encoding="utf-8").splitlines()[0].strip()
            if tok:
                return tok
        except Exception:
            continue
    return None


def maybe_token_kwargs() -> dict:
    tok = resolve_hf_token()
    return {"token": tok} if tok else {}


def _tokenize_for_rouge(text: str) -> List[str]:
    # 新闻生成任务：做一个轻量 tokenization（只用于 ROUGE 近似）
    # - lower
    # - 保留字母/数字，其他当分隔符
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0 or len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _f1_overlap(pred_counts: Dict[Tuple[str, ...], int], ref_counts: Dict[Tuple[str, ...], int]) -> float:
    if not pred_counts or not ref_counts:
        return 0.0
    overlap = 0
    pred_total = 0
    ref_total = 0
    for k, v in pred_counts.items():
        pred_total += v
        overlap += min(v, ref_counts.get(k, 0))
    for v in ref_counts.values():
        ref_total += v
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    p = overlap / pred_total
    r = overlap / ref_total
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _lcs_len(a: List[str], b: List[str]) -> int:
    # 标准 DP：O(len(a)*len(b))，eval_size 不大时足够
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]


def rouge_f1(pred: str, ref: str) -> Dict[str, float]:
    pred_toks = _tokenize_for_rouge(pred)
    ref_toks = _tokenize_for_rouge(ref)
    r1 = _f1_overlap(_ngram_counts(pred_toks, 1), _ngram_counts(ref_toks, 1))
    r2 = _f1_overlap(_ngram_counts(pred_toks, 2), _ngram_counts(ref_toks, 2))
    lcs = _lcs_len(pred_toks, ref_toks)
    if not pred_toks or not ref_toks or lcs == 0:
        rl = 0.0
    else:
        p = lcs / len(pred_toks)
        r = lcs / len(ref_toks)
        rl = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"rouge1_f1": r1, "rouge2_f1": r2, "rougeL_f1": rl}


def load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_prompt(item: dict) -> str:
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    return instruction + (("\n" + inp) if inp else "")


def resolve_base_model(adapter_dir: str, base_model_arg: str | None) -> str:
    if base_model_arg:
        return base_model_arg
    meta = Path(adapter_dir) / "training_metadata.json"
    if meta.exists():
        try:
            j = json.loads(meta.read_text(encoding="utf-8"))
            bm = j.get("base_model")
            if isinstance(bm, str) and bm.strip():
                return bm.strip()
        except Exception:
            pass
    # 兜底：尽量不猜，提示用户显式给
    raise ValueError("无法从 training_metadata.json 推断 base_model，请传 --base_model")


def load_model_and_tokenizer(base_model: str, adapter_dir: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True, **maybe_token_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        **maybe_token_kwargs(),
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def load_base_only(base_model: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, **maybe_token_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        **maybe_token_kwargs(),
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **({"temperature": temperature, "top_p": top_p} if do_sample else {}),
    )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # 去掉 prompt 前缀，避免评测把提示词也算进去
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="SFT LoRA 评测（ROUGE，无额外依赖）")
    ap.add_argument("--adapter_dir", type=str, required=True, help="训练输出目录：包含 adapter_model.safetensors 等")
    ap.add_argument("--data_path", type=str, default="project/data/processed/sft_data.jsonl")
    ap.add_argument("--base_model", type=str, default=None, help="不传则尝试从 training_metadata.json 推断")
    ap.add_argument("--no_lora", action="store_true", help="只评测 base_model（不加载 LoRA），用于做 before/after 对比")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_size", type=int, default=200, help="评测样本数（从全量数据随机抽样）")
    ap.add_argument("--num_print", type=int, default=5, help="打印多少条对比样例")
    ap.add_argument("--cpu_threads", type=int, default=0, help="CPU 推理线程数（0 表示不设置）")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "bf16"], help="权重 dtype；无卡建议 fp32（稳）或 bf16（省内存，视 CPU 支持）")

    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true", help="开启采样（默认关闭，评测更稳定）")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    args = ap.parse_args()

    random.seed(args.seed)
    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    adapter_dir = args.adapter_dir
    if not os.path.exists(adapter_dir):
        raise FileNotFoundError(f"adapter_dir 不存在：{adapter_dir}")

    base_model = resolve_base_model(adapter_dir, args.base_model)
    print(f"🔧 base_model: {base_model}")
    print(f"🔧 adapter_dir: {adapter_dir}")
    print(f"📥 loading data: {args.data_path}")
    items = load_jsonl(args.data_path)
    if not items:
        raise ValueError("数据为空")

    eval_size = min(max(args.eval_size, 1), len(items))
    eval_items = random.sample(items, eval_size)
    print(f"🧪 eval_size: {eval_size} / {len(items)}")

    # dtype 选择（主要给无卡节省内存用）
    if args.dtype != "auto":
        if args.dtype == "fp32":
            torch.set_default_dtype(torch.float32)
        elif args.dtype == "bf16":
            torch.set_default_dtype(torch.bfloat16)

    if args.no_lora:
        print("🧱 模式：base_model only（no_lora）")
        model, tokenizer = load_base_only(base_model)
    else:
        print("🧩 模式：base_model + LoRA adapter")
        model, tokenizer = load_model_and_tokenizer(base_model, adapter_dir)

    sums = {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
    shown = 0
    for idx, it in enumerate(eval_items, start=1):
        prompt = build_prompt(it)
        ref = (it.get("output") or "").strip()
        pred = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        m = rouge_f1(pred, ref)
        for k in sums:
            sums[k] += m[k]

        if shown < args.num_print:
            shown += 1
            print("\n" + "=" * 88)
            print(f"[{shown}/{args.num_print}] 样例 idx={idx}")
            print("- prompt (instruction+summary) -")
            print(prompt)
            print("\n- pred -")
            print(pred)
            print("\n- ref -")
            print(ref)
            print("\n- rouge -")
            print({k: round(v, 4) for k, v in m.items()})

    avg = {k: (sums[k] / eval_size) for k in sums}
    print("\n" + "#" * 88)
    print("✅ ROUGE (F1) 平均值：")
    for k in ["rouge1_f1", "rouge2_f1", "rougeL_f1"]:
        print(f"{k}: {avg[k]:.4f}")
    print("#" * 88)


if __name__ == "__main__":
    main()


