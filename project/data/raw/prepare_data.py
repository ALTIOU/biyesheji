from datasets import load_dataset
import random
import json
import os
import re

# =============================
# 配置参数
# =============================

SFT_SIZE = 3000         # number of SFT samples
RL_PROMPT_SIZE = 300    # number of RL prompts
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


# =============================
# 第 1 步：下载 XSum 数据集
# =============================

print("Downloading XSum dataset...")
dataset = load_dataset("xsum", trust_remote_code=True)

train_data = dataset["train"]
print(f"Train split size: {len(train_data)}")


# =============================
# 第 2 步：随机打乱
# =============================

print("Shuffling data...")
train_list = list(train_data)
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


documents = [clean(item["document"]) for item in train_list]
summaries = [clean(item["summary"]) for item in train_list]


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
# 第 6 步：制作测试集（人类文本）
# =============================

print("Generating test set (Human)...")

test_human = documents[SFT_SIZE:SFT_SIZE + TEST_HUMAN_SIZE]

with open(f"{OUTPUT_DIR}/test_human.jsonl", "w", encoding="utf-8") as f:
    for t in test_human:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

print(f"✔ Human 测试集已生成：{OUTPUT_DIR}/test_human.jsonl")


# =============================
# 第 7 步：制作测试集（AI 文本）
# 这里先留空，由你的生成模型后续生成
# =============================

print("Creating test set placeholder (AI)...")

with open(f"{OUTPUT_DIR}/test_ai_placeholder.jsonl", "w", encoding="utf-8") as f:
    # 保持 jsonl 格式：后续你生成 AI 文本时，每行写入 {"text": "..."} 即可
    pass

print(f"Wrote: {OUTPUT_DIR}/test_ai_placeholder.jsonl")


print("\nDone. Output files:")
print(f"{OUTPUT_DIR}/")
print("├── sft_data.jsonl")
print("├── rl_prompts.jsonl")
print("├── test_human.jsonl")
print("└── test_ai_placeholder.jsonl")