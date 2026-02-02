"""
测试最佳RL模型（Prompt 12，Reward=0.6772）
"""
import sys
sys.path.append('biyesheji/project/src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载模型
base_model_path = "Qwen/Qwen3-7B"
rl_adapter_path = "biyesheji/project/models/rl_detectgpt_pure_exp2_scaled/grpo_detectgpt_pure_scaled_20260125"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading RL adapter...")
model = PeftModel.from_pretrained(model, rl_adapter_path)
model = model.merge_and_unload()  # 合并LoRA权重
model.eval()

# 测试prompts
test_prompts = [
    "人工智能的未来发展趋势是什么？",
    "如何提高工作效率？",
    "解释一下量子计算的基本原理。",
]

print("\n" + "="*60)
print("测试RL优化后的模型生成")
print("="*60 + "\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"[Prompt {i}] {prompt}")
    print("-" * 60)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    print(f"[Response]\n{response}\n")
    print("="*60 + "\n")
