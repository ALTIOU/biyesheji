"""
改进的模型对比脚本
修复：
1. 增加max_tokens到300，保证完整生成
2. 检测文本是否被截断
3. 保存为JSON避免编码问题
"""
import sys
sys.path.append('biyesheji/project/src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model(base_model_path, adapter_path=None):
    """加载模型"""
    print(f"Loading model from {base_model_path}...")
    if adapter_path:
        print(f"  with adapter: {adapter_path}")
    
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

def generate_text(model, tokenizer, prompt, max_tokens=300):
    """
    生成文本（增加到300 tokens）
    
    Returns:
        dict: {
            "text": 生成的文本,
            "is_complete": 是否完整生成（没被截断）,
            "actual_tokens": 实际生成的token数
        }
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    # 检查是否被截断（生成的tokens达到max_tokens）
    generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
    is_complete = generated_tokens < max_tokens
    
    return {
        "text": response,
        "is_complete": is_complete,
        "actual_tokens": generated_tokens,
        "max_tokens": max_tokens
    }

def main():
    base_model_path = "Qwen/Qwen3-7B"
    sft_adapter_path = "biyesheji/project/models/sft/sft_20260105_1453"
    rl_adapter_path = "biyesheji/project/models/rl_detectgpt_pure_exp2_scaled/grpo_detectgpt_pure_scaled_20260125"
    
    test_prompts = [
        "人工智能的未来发展趋势是什么？",
        "如何提高工作效率？",
        "量子计算的基本原理是什么？",
    ]
    
    print("="*80)
    print("Improved Model Comparison (max_tokens=300)")
    print("="*80)
    
    print("\n[1/2] Loading SFT model...")
    sft_tokenizer, sft_model = load_model(base_model_path, sft_adapter_path)
    
    print("\n[2/2] Loading RL model...")
    rl_tokenizer, rl_model = load_model(base_model_path, rl_adapter_path)
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"[Prompt {i}/{len(test_prompts)}] {prompt}")
        print('='*80)
        
        # SFT生成
        print("\n[Generating with SFT model...]")
        sft_result = generate_text(sft_model, sft_tokenizer, prompt, max_tokens=300)
        print(f"  Tokens: {sft_result['actual_tokens']}/{sft_result['max_tokens']}")
        print(f"  Complete: {'Yes' if sft_result['is_complete'] else 'TRUNCATED!'}")
        print(f"  Length: {len(sft_result['text'])} chars")
        
        # RL生成
        print("\n[Generating with RL model...]")
        rl_result = generate_text(rl_model, rl_tokenizer, prompt, max_tokens=300)
        print(f"  Tokens: {rl_result['actual_tokens']}/{rl_result['max_tokens']}")
        print(f"  Complete: {'Yes' if rl_result['is_complete'] else 'TRUNCATED!'}")
        print(f"  Length: {len(rl_result['text'])} chars")
        
        results.append({
            "prompt": prompt,
            "sft": {
                "text": sft_result["text"],
                "is_complete": sft_result["is_complete"],
                "tokens": sft_result["actual_tokens"],
                "chars": len(sft_result["text"])
            },
            "rl": {
                "text": rl_result["text"],
                "is_complete": rl_result["is_complete"],
                "tokens": rl_result["actual_tokens"],
                "chars": len(rl_result["text"])
            }
        })
        
        print(f"\n[Statistics]")
        print(f"  SFT: {len(sft_result['text'])} chars, {'complete' if sft_result['is_complete'] else 'TRUNCATED'}")
        print(f"  RL:  {len(rl_result['text'])} chars, {'complete' if rl_result['is_complete'] else 'TRUNCATED'}")
    
    # 保存结果
    output_file = "biyesheji/project/outputs/model_comparison_improved.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print('='*80)
    
    # 统计
    print("\n[Overall Statistics]")
    sft_truncated = sum(1 for r in results if not r["sft"]["is_complete"])
    rl_truncated = sum(1 for r in results if not r["rl"]["is_complete"])
    
    print(f"SFT model: {sft_truncated}/{len(results)} truncated")
    print(f"RL model:  {rl_truncated}/{len(results)} truncated")
    
    avg_sft_len = sum(r["sft"]["chars"] for r in results) / len(results)
    avg_rl_len = sum(r["rl"]["chars"] for r in results) / len(results)
    
    print(f"\nAverage length:")
    print(f"  SFT: {avg_sft_len:.0f} chars")
    print(f"  RL:  {avg_rl_len:.0f} chars")

if __name__ == "__main__":
    main()
