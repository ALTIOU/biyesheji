"""
从XSum数据集重新提取test_human，同时包含prompt和text
这样就可以用于评估：给定prompt，对比人类回答vs AI生成
"""
import random
import json
import os
import re

# 配置
TEST_HUMAN_SIZE = 200
SFT_SIZE = 3000
EVAL_SFT_SIZE = 10

OUTPUT_DIR = "biyesheji/project/data/processed"

# CJK检测
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))

def clean(text):
    """去除换行和多余空格"""
    return text.replace("\n", " ").strip()

# 加载XSum数据集
print("Loading XSum dataset...")
try:
    from datasets import load_dataset
    try:
        ds = load_dataset("xsum")
    except ValueError as e:
        if "trust_remote_code" in str(e):
            ds = load_dataset("xsum", trust_remote_code=True)
        else:
            raise
    
    train_list = list(ds["train"])
    print(f"Loaded {len(train_list)} train samples")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using fallback method...")
    # Fallback方法（如果需要）
    raise

# 随机打乱
random.seed(42)  # 使用相同的seed保持一致性
random.shuffle(train_list)

# 过滤CJK
filtered = []
for item in train_list:
    doc = clean(item["document"])
    summ = clean(item["summary"])
    if has_cjk(doc) or has_cjk(summ):
        continue
    filtered.append({
        "text": doc,      # 完整文章
        "prompt": summ    # 摘要作为prompt
    })

print(f"Filtered: {len(filtered)} samples (removed CJK)")

# 计算test_human的起始位置（与prepare_data.py保持一致）
test_start = SFT_SIZE + EVAL_SFT_SIZE
test_end = test_start + TEST_HUMAN_SIZE

if len(filtered) < test_end:
    raise RuntimeError(f"Not enough samples: need {test_end}, got {len(filtered)}")

# 提取test_human（同时包含prompt和text）
test_human_with_prompt = filtered[test_start:test_end]

# 保存
output_file = f"{OUTPUT_DIR}/test_human_with_prompt.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in test_human_with_prompt:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Saved {len(test_human_with_prompt)} samples to:")
print(f"   {output_file}")

# 打印样例
print("\n[Sample 1]")
print(f"Prompt: {test_human_with_prompt[0]['prompt'][:100]}...")
print(f"Text:   {test_human_with_prompt[0]['text'][:100]}...")

# 统计
avg_prompt_len = sum(len(item['prompt']) for item in test_human_with_prompt) / len(test_human_with_prompt)
avg_text_len = sum(len(item['text']) for item in test_human_with_prompt) / len(test_human_with_prompt)

print(f"\n[Statistics]")
print(f"Total samples: {len(test_human_with_prompt)}")
print(f"Avg prompt length: {avg_prompt_len:.0f} chars")
print(f"Avg text length: {avg_text_len:.0f} chars")
