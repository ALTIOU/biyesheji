"""
对比SFT模型和RL优化后模型的生成质量
"""
import sys
sys.path.append('biyesheji/project/src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model(base_model_path, adapter_path=None):
    """加载模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    return tokenizer, model

def generate_text(model, tokenizer, prompt, max_tokens=150):
    """生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    return response

def main():
    base_model_path = "Qwen/Qwen3-7B"
    sft_adapter_path = "biyesheji/project/models/sft/sft_20260105_1453"
    rl_adapter_path = "biyesheji/project/models/rl_detectgpt_pure_exp2_scaled/grpo_detectgpt_pure_scaled_20260125"
    
    test_prompts = [
        "人工智能的未来发展趋势是什么？",
        "如何提高工作效率？",
        "量子计算的基本原理是什么？",
    ]
    
    print("Loading SFT model...")
    sft_tokenizer, sft_model = load_model(base_model_path, sft_adapter_path)
    
    print("Loading RL model...")
    rl_tokenizer, rl_model = load_model(base_model_path, rl_adapter_path)
    
    results = []
    
    print("\n" + "="*80)
    print("Comparing SFT vs RL models")
    print("="*80 + "\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[Prompt {i}] {prompt}")
        print("-" * 80)
        
        # SFT生成
        print("\n[SFT Model]")
        sft_response = generate_text(sft_model, sft_tokenizer, prompt)
        print(sft_response)
        
        # RL生成
        print("\n[RL Model]")
        rl_response = generate_text(rl_model, rl_tokenizer, prompt)
        print(rl_response)
        
        print("\n" + "="*80)
        
        results.append({
            "prompt": prompt,
            "sft_response": sft_response,
            "rl_response": rl_response
        })
    
    # 保存结果为JSON（避免编码问题）
    output_file = "biyesheji/project/outputs/model_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
