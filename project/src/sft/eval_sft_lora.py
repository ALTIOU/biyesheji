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


def load_prompts(prompts_path: str) -> List[dict]:
    """
    支持两种格式：
    - .txt：每行一个 prompt
    - .jsonl：每行一个 JSON，优先取 prompt 字段，否则按 instruction/input 拼接

    返回：[{id, prompt, reference?}...]
    """
    p = Path(prompts_path)
    if not p.exists():
        raise FileNotFoundError(f"prompts_path 不存在：{prompts_path}")

    items: List[dict] = []
    if p.suffix.lower() == ".txt":
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            s = line.strip()
            if not s:
                continue
            items.append({"id": str(i), "prompt": s})
        return items

    if p.suffix.lower() == ".jsonl":
        raw = load_jsonl(str(p))
        for i, it in enumerate(raw, start=1):
            if not isinstance(it, dict):
                continue
            prompt = (it.get("prompt") or "").strip()
            if not prompt:
                prompt = build_prompt(it).strip()
            if not prompt:
                continue
            rid = str(it.get("id") or i)
            out = {"id": rid, "prompt": prompt}
            if "reference" in it and isinstance(it.get("reference"), str):
                out["reference"] = it["reference"].strip()
            elif "output" in it and isinstance(it.get("output"), str):
                out["reference"] = it["output"].strip()
            items.append(out)
        return items

    raise ValueError("prompts_path 仅支持 .txt 或 .jsonl")


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def check_prompts_unseen(prompts: List[dict], data_path: str) -> List[dict]:
    """
    粗略检查：prompt 是否与训练/数据集里的 prompt 完全一致（做了空白归一化）。
    返回命中的 prompts 列表（可能为空）。
    """
    try:
        items = load_jsonl(data_path)
    except Exception:
        return []

    seen = set()
    for it in items:
        try:
            seen.add(normalize_text(build_prompt(it)))
        except Exception:
            continue

    hits = []
    for p in prompts:
        if normalize_text(p.get("prompt", "")) in seen:
            hits.append(p)
    return hits


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


def load_tokenizer_for_compare(base_model: str, adapter_dir: str):
    """
    对比模式尽量用同一个 tokenizer：优先用 adapter_dir（训练时保存的 tokenizer），没有则用 base_model。
    """
    src = adapter_dir if Path(adapter_dir).exists() else base_model
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, **maybe_token_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def save_jsonl(path: str, rows: List[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_markdown_compare(path: str, rows: List[dict], title: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    for r in rows:
        rid = r.get("id", "")
        prompt = r.get("prompt", "")
        base_out = r.get("base_output", "")
        sft_out = r.get("sft_output", "")
        lines.append(f"## id={rid}")
        lines.append("")
        lines.append("### Prompt")
        lines.append("```")
        lines.append(str(prompt))
        lines.append("```")
        lines.append("")
        lines.append("### Base 输出")
        lines.append("```")
        lines.append(str(base_out))
        lines.append("```")
        lines.append("")
        lines.append("### SFT-LoRA 输出")
        lines.append("```")
        lines.append(str(sft_out))
        lines.append("```")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def nll_per_token(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    max_length: int = 2048,
) -> float:
    """
    计算 teacher-forcing 的平均 NLL（per-token loss），只对 reference 部分计入 loss。
    这能更直接衡量“给定 summary 是否更会写正文”，比单次采样生成更稳定。

    注意：为避免 OOM/过长，这里对 prompt+reference 做截断。
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False).get("input_ids", [])
    ref_ids = tokenizer(reference, add_special_tokens=False).get("input_ids", [])
    if not ref_ids:
        return float("nan")

    input_ids = (prompt_ids + ref_ids)[:max_length]
    # labels：prompt 部分不计 loss
    labels = ([-100] * min(len(prompt_ids), max_length) + ref_ids)[:max_length]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    labels_t = torch.tensor([labels], dtype=torch.long)
    attn = torch.ones_like(input_ids_t)

    if torch.cuda.is_available():
        input_ids_t = input_ids_t.to(model.device)
        labels_t = labels_t.to(model.device)
        attn = attn.to(model.device)

    out = model(input_ids=input_ids_t, attention_mask=attn, labels=labels_t)
    loss = out.loss
    return float(loss.item()) if loss is not None else float("nan")


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

    ap.add_argument("--prompts_path", type=str, default=None, help="固定 prompts 文件（.txt 每行一个 或 .jsonl），用于生成示例对比")
    ap.add_argument("--compare_base", action="store_true", help="固定 prompts 下同时生成 base 和 sft（先 base 再加载 LoRA）")
    ap.add_argument("--check_unseen", action="store_true", help="在 compare/prompts 模式下，检查 prompts 是否与 data_path 的 prompt 完全一致")
    ap.add_argument("--out_jsonl", type=str, default=None, help="prompts 对比输出 jsonl 路径（仅 prompts/compare 模式）")
    ap.add_argument("--out_md", type=str, default=None, help="prompts 对比输出 markdown 路径（仅 prompts/compare 模式）")
    ap.add_argument("--eval_no_sample", action="store_true", help="ROUGE eval 时不 random.sample，按文件顺序取前 eval_size 条（做固定集更稳定）")
    ap.add_argument("--ppl_eval", action="store_true", help="在 prompts/compare 下额外计算 teacher-forcing NLL/PPL（更稳的能力对比）")
    ap.add_argument("--ppl_max_length", type=int, default=2048, help="PPL/NLL 评估时的最大 token 长度（prompt+ref 截断）")

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

    # dtype 选择（主要给无卡节省内存用）
    if args.dtype != "auto":
        if args.dtype == "fp32":
            torch.set_default_dtype(torch.float32)
        elif args.dtype == "bf16":
            torch.set_default_dtype(torch.bfloat16)

    # 固定 prompts：用于论文前后对比展示
    if args.prompts_path:
        prompts = load_prompts(args.prompts_path)
        if not prompts:
            raise ValueError("prompts_path 读取为空，请检查文件内容")

        if args.check_unseen:
            hits = check_prompts_unseen(prompts, args.data_path)
            if hits:
                print("⚠️ 发现 prompts 与 data_path 中的 prompt 完全一致（可能不算未见过）：")
                for h in hits[:20]:
                    print(f"- id={h.get('id')} prompt={h.get('prompt')[:80]}")
                print("建议替换这些 prompts 后再跑。")

        if args.compare_base:
            print(f"🧪 模式：固定 prompts + base vs SFT-LoRA 对比，共 {len(prompts)} 条")
            tokenizer = load_tokenizer_for_compare(base_model, adapter_dir)
            base, _ = load_base_only(base_model)

            rows: List[dict] = []
            # 先跑 base
            base_sums = {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
            base_cnt = 0
            base_nll_sum = 0.0
            base_nll_cnt = 0
            for p in prompts:
                prompt = p["prompt"]
                base_out = generate_one(
                    model=base,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                row = {"id": p.get("id"), "prompt": prompt, "base_output": base_out}
                if "reference" in p and isinstance(p.get("reference"), str) and p["reference"].strip():
                    row["reference"] = p["reference"].strip()
                    m = rouge_f1(base_out, row["reference"])
                    row["base_rouge"] = m
                    for k in base_sums:
                        base_sums[k] += m[k]
                    base_cnt += 1
                    if args.ppl_eval:
                        nll = nll_per_token(
                            model=base,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            reference=row["reference"],
                            max_length=args.ppl_max_length,
                        )
                        row["base_nll"] = nll
                        if nll == nll:  # not nan
                            base_nll_sum += nll
                            base_nll_cnt += 1
                            row["base_ppl"] = float(torch.exp(torch.tensor(nll)).item())
                rows.append(row)

            # 再加载 LoRA 并跑 SFT
            sft = PeftModel.from_pretrained(base, adapter_dir)
            sft.eval()
            sft_sums = {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
            sft_cnt = 0
            sft_nll_sum = 0.0
            sft_nll_cnt = 0
            for r in rows:
                sft_out = generate_one(
                    model=sft,
                    tokenizer=tokenizer,
                    prompt=r["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                r["sft_output"] = sft_out
                if "reference" in r and isinstance(r.get("reference"), str) and r["reference"].strip():
                    m = rouge_f1(sft_out, r["reference"])
                    r["sft_rouge"] = m
                    for k in sft_sums:
                        sft_sums[k] += m[k]
                    sft_cnt += 1
                    if args.ppl_eval:
                        nll = nll_per_token(
                            model=sft,
                            tokenizer=tokenizer,
                            prompt=r["prompt"],
                            reference=r["reference"],
                            max_length=args.ppl_max_length,
                        )
                        r["sft_nll"] = nll
                        if nll == nll:
                            sft_nll_sum += nll
                            sft_nll_cnt += 1
                            r["sft_ppl"] = float(torch.exp(torch.tensor(nll)).item())

            if args.out_jsonl:
                save_jsonl(args.out_jsonl, rows)
                print(f"💾 已保存：{args.out_jsonl}")
            if args.out_md:
                save_markdown_compare(args.out_md, rows, title="SFT 前后对比（固定 prompts）")
                print(f"💾 已保存：{args.out_md}")

            if base_cnt > 0 and sft_cnt > 0:
                base_avg = {k: base_sums[k] / base_cnt for k in base_sums}
                sft_avg = {k: sft_sums[k] / sft_cnt for k in sft_sums}
                print("\n" + "#" * 88)
                print(f"✅ 固定集 ROUGE (F1) 平均值（N={base_cnt}，仅对含 reference 的样本计算）：")
                print("base:", {k: round(v, 4) for k, v in base_avg.items()})
                print("sft :", {k: round(v, 4) for k, v in sft_avg.items()})
                if args.ppl_eval and base_nll_cnt > 0 and sft_nll_cnt > 0:
                    base_nll_avg = base_nll_sum / base_nll_cnt
                    sft_nll_avg = sft_nll_sum / sft_nll_cnt
                    print(f"✅ NLL/PPL（teacher-forcing，越低越好；N={base_nll_cnt}）：")
                    print("base:", {"nll": round(base_nll_avg, 4), "ppl": round(float(torch.exp(torch.tensor(base_nll_avg)).item()), 4)})
                    print("sft :", {"nll": round(sft_nll_avg, 4), "ppl": round(float(torch.exp(torch.tensor(sft_nll_avg)).item()), 4)})
                print("#" * 88)

            # 控制台也打印几条，方便快速看
            for r in rows[: min(args.num_print, len(rows))]:
                print("\n" + "=" * 88)
                print(f"id={r.get('id')}")
                print("- prompt -")
                print(r.get("prompt"))
                print("\n- base -")
                print(r.get("base_output"))
                print("\n- sft -")
                print(r.get("sft_output"))
                if "reference" in r:
                    print("\n- ref -")
                    print(r.get("reference"))
                if "base_rouge" in r or "sft_rouge" in r:
                    print("\n- rouge -")
                    if "base_rouge" in r:
                        print("base:", {k: round(v, 4) for k, v in r["base_rouge"].items()})
                    if "sft_rouge" in r:
                        print("sft :", {k: round(v, 4) for k, v in r["sft_rouge"].items()})
            return

        # 仅 prompts：按 no_lora 决定用 base 或 sft
        print(f"🧪 模式：固定 prompts 生成（共 {len(prompts)} 条），no_lora={args.no_lora}")
        tokenizer = load_tokenizer_for_compare(base_model, adapter_dir)
        if args.no_lora:
            model, _ = load_base_only(base_model)
        else:
            base, _ = load_base_only(base_model)
            model = PeftModel.from_pretrained(base, adapter_dir)
            model.eval()

        rows: List[dict] = []
        for p in prompts:
            pred = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=p["prompt"],
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = {"id": p.get("id"), "prompt": p["prompt"], "pred": pred}
            if "reference" in p:
                row["reference"] = p["reference"]
            rows.append(row)

        if args.out_jsonl:
            save_jsonl(args.out_jsonl, rows)
            print(f"💾 已保存：{args.out_jsonl}")
        for r in rows[: min(args.num_print, len(rows))]:
            print("\n" + "=" * 88)
            print(f"id={r.get('id')}")
            print("- prompt -")
            print(r.get("prompt"))
            print("\n- pred -")
            print(r.get("pred"))
        return

    # 默认：走原来的 ROUGE eval（随机抽样 data_path）
    print(f"📥 loading data: {args.data_path}")
    items = load_jsonl(args.data_path)
    if not items:
        raise ValueError("数据为空")

    eval_size = min(max(args.eval_size, 1), len(items))
    if args.eval_no_sample:
        eval_items = items[:eval_size]
    else:
        eval_items = random.sample(items, eval_size)
    print(f"🧪 eval_size: {eval_size} / {len(items)}")

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


