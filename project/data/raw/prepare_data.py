import random
import json
import os
import re

# =============================
# 配置参数
# =============================

SFT_SIZE = 3000         # number of SFT samples
RL_PROMPT_SIZE = 300    # number of RL prompts
EVAL_SFT_SIZE = 10      # fixed eval set size (unseen during SFT)
TEST_HUMAN_SIZE = 200   # test set: human texts
TEST_AI_SIZE = 200      # test set: AI texts (generated later)

# Always write to project/data/processed regardless of current working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "processed"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Language policy (EN-only)
# =============================
SFT_INSTRUCTION = "Write a news report in English based on the following summary:"
RL_PROMPT_TEMPLATE = 'Write a short news report in English about "{topic}".'
TOPICS = [
    "AI regulation",
    "renewable energy technology",
    "space exploration",
    "global economic outlook",
    "climate change policy",
    "medical technology breakthroughs",
    "cybersecurity",
    "education technology reform",
    "social hot issues",
    "international relations and diplomacy",
]

# Guardrail: fail fast if any CJK characters appear in generated data
# 保证所有写进实验数据的文本都是纯英文
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def assert_no_cjk(text: str, field: str) -> None:
    if _CJK_RE.search(text or ""):
        raise ValueError(f"Found CJK characters in {field}: {text[:80]!r}")


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


# =============================
# 第 1 步：下载 XSum 数据集
# =============================

print("Downloading XSum dataset...")
def load_xsum_train_rows(limit: int):
    """
    兼容性加载：
    - 优先 datasets.load_dataset("xsum")
    - 若 datasets 在当前环境不可用/报错，则 fallback：huggingface-hub 拉 parquet 并用 pyarrow 读取
    只取 train split 的前 limit 条（随后会 random.shuffle）。
    """
    # 1) 优先 datasets
    try:
        from datasets import load_dataset  # type: ignore

        # 说明：部分 datasets 版本需要 trust_remote_code=True 才能加载 xsum（自定义脚本）
        try:
            ds = load_dataset("xsum")
        except ValueError as e:
            msg = str(e)
            if "trust_remote_code=True" in msg or "trust_remote_code" in msg:
                try:
                    ds = load_dataset("xsum", trust_remote_code=True)
                except TypeError:
                    # 某些版本不再支持该参数，就把原异常抛回
                    raise
            else:
                raise
        train = ds["train"]
        rows = []
        for i, ex in enumerate(train):
            rows.append({"document": ex["document"], "summary": ex["summary"]})
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            return rows
        # 不够也返回，让后续 fallback 补齐
    except Exception as e:
        print(f"[WARN] datasets.load_dataset('xsum') 失败，改用 parquet fallback：{type(e).__name__}: {e}")

    # 2) fallback：直接从 HF datasets 仓库读取 parquet
    from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    files = [f for f in list_repo_files("xsum", repo_type="dataset") if f.startswith("train") and f.endswith(".parquet")]
    files.sort()
    if not files:
        raise RuntimeError("未在 HuggingFace 数据集仓库中找到 train*.parquet 文件，无法 fallback 加载 XSum。")

    rows = []
    for fn in files:
        local_path = hf_hub_download(repo_id="xsum", repo_type="dataset", filename=fn)
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=2048, columns=["document", "summary"]):
            docs = batch.column(0).to_pylist()
            sums = batch.column(1).to_pylist()
            for d, s in zip(docs, sums):
                rows.append({"document": d, "summary": s})
                if len(rows) >= limit:
                    return rows
    return rows


NEED_ROWS = SFT_SIZE + EVAL_SFT_SIZE + TEST_HUMAN_SIZE
# XSum 偶尔会包含少量 CJK 字符（例如引用/人名/地名），为了保持 EN-only 约束，
# 我们多取一些样本，再在本地过滤掉含 CJK 的样本。
NEED_RAW_ROWS = NEED_ROWS + 500
train_list = load_xsum_train_rows(NEED_RAW_ROWS)
if len(train_list) < NEED_RAW_ROWS:
    raise RuntimeError(f"XSum train 样本不足：需要至少 {NEED_RAW_ROWS}，实际拿到 {len(train_list)}")

print(f"Loaded train rows: {len(train_list)} (need {NEED_ROWS}, raw {NEED_RAW_ROWS})")


# =============================
# 第 2 步：随机打乱
# =============================

print("Shuffling data...")
random.shuffle(train_list)


# =============================
# 第 3 步：提取所需字段
# 每条数据包含：
# - 'document'：新闻正文（我们要的）
# - 'summary'：摘要（作为提示 input，更符合 instruction-following）
# =============================

def clean(text):
    """Simple cleanup: remove newlines and extra spaces."""
    return text.replace("\n", " ").strip()


filtered = []
for item in train_list:
    doc = clean(item["document"])
    summ = clean(item["summary"])
    if has_cjk(doc) or has_cjk(summ):
        continue
    filtered.append((doc, summ))

if len(filtered) < NEED_ROWS:
    raise RuntimeError(
        f"过滤 CJK 后样本不足：需要 {NEED_ROWS}，实际 {len(filtered)}。"
        f"请把 NEED_RAW_ROWS 调大（当前 {NEED_RAW_ROWS}）。"
    )

documents = [d for d, _ in filtered]
summaries = [s for _, s in filtered]


# =============================
# 第 4 步：制作 SFT 数据集
# 格式：
# {"instruction": "...", "output": "..."}
# =============================

print("Generating SFT dataset...")

sft_data = []
for i in range(SFT_SIZE):
    assert_no_cjk(summaries[i], "sft.input")
    assert_no_cjk(documents[i], "sft.output")
    sft_data.append({
        "instruction": SFT_INSTRUCTION,
        "input": summaries[i],
        "output": documents[i],
    })

with open(f"{OUTPUT_DIR}/sft_data.jsonl", "w", encoding="utf-8") as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Wrote: {OUTPUT_DIR}/sft_data.jsonl")


# =============================
# 第 5 步：生成 RL Prompt 数据
# 只生成 prompt，不包含输出
# =============================

print("Generating RL prompts...")

def make_prompt():
    t = random.choice(TOPICS)
    return RL_PROMPT_TEMPLATE.format(topic=t)

rl_prompts = [{"prompt": make_prompt()} for _ in range(RL_PROMPT_SIZE)]
for item in rl_prompts:
    assert_no_cjk(item["prompt"], "rl.prompt")

with open(f"{OUTPUT_DIR}/rl_prompts.jsonl", "w", encoding="utf-8") as f:
    for item in rl_prompts:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Wrote: {OUTPUT_DIR}/rl_prompts.jsonl")


# =============================
# 第 6 步：制作 Eval 集（与 SFT 同格式，但来自未见过样本）
# =============================

print("Generating eval set (SFT-format, unseen)...")

eval_start = SFT_SIZE
eval_end = SFT_SIZE + EVAL_SFT_SIZE

eval_sft = []
for i in range(eval_start, eval_end):
    assert_no_cjk(summaries[i], "eval_sft.input")
    assert_no_cjk(documents[i], "eval_sft.output")
    eval_sft.append({
        "instruction": SFT_INSTRUCTION,
        "input": summaries[i],
        "output": documents[i],
    })

with open(f"{OUTPUT_DIR}/eval_sft_10.jsonl", "w", encoding="utf-8") as f:
    for item in eval_sft:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[OK] Eval set written: {OUTPUT_DIR}/eval_sft_10.jsonl")


# =============================
# 第 7 步：制作测试集（人类文本，仅正文，用于检测任务）
# =============================

print("Generating test set (Human, document-only)...")

test_start = eval_end
test_end = eval_end + TEST_HUMAN_SIZE
test_human = documents[test_start:test_end]

with open(f"{OUTPUT_DIR}/test_human.jsonl", "w", encoding="utf-8") as f:
    for t in test_human:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

print(f"[OK] Human test set written: {OUTPUT_DIR}/test_human.jsonl")


# =============================
# 第 8 步：制作测试集（AI 文本）
# 这里先留空，由你的生成模型后续生成
# =============================

print("Creating test set placeholder (AI)...")

with open(f"{OUTPUT_DIR}/test_ai_placeholder.jsonl", "w", encoding="utf-8") as f:
    # 保持 jsonl 格式：后续你生成 AI 文本时，每行写入 {"text": "..."} 即可
    pass

print(f"Wrote: {OUTPUT_DIR}/test_ai_placeholder.jsonl")


print("\nDone. Output files:")
print(f"{OUTPUT_DIR}/")
print("- sft_data.jsonl")
print("- rl_prompts.jsonl")
print("- eval_sft_10.jsonl")
print("- test_human.jsonl")
print("- test_ai_placeholder.jsonl")