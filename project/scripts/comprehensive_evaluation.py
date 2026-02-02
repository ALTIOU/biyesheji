"""
全面评估脚本
对比：人类文本 vs SFT模型 vs RL模型

改进：
1. 使用test_human_with_prompt（包含prompt-text配对）
2. max_tokens=300，保证完整生成
3. 多维度评估（DetectGPT + 其他指标）
"""
import sys
import os

# 设置正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
src_dir = os.path.join(project_dir, "src")
sys.path.insert(0, src_dir)

os.environ["USE_T5_PERTURBATION"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rl.reward_functions import DetectGPTDetector
import json

def load_model(base_model_path, adapter_path=None):
    """加载模型"""
    print(f"Loading {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if adapter_path:
        print(f"  Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    return tokenizer, model

def generate_text(model, tokenizer, prompt, max_tokens=300):
    """生成文本"""
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
    
    # 检查是否完整
    generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
    is_complete = generated_tokens < max_tokens
    
    return response, is_complete, generated_tokens

def main():
    print("="*80)
    print("Comprehensive Evaluation: Human vs SFT vs RL")
    print("="*80)
    
    # 1. 加载test_human数据（带prompt）
    print("\n[Step 1] Loading test_human_with_prompt.jsonl...")
    test_file = os.path.join(project_dir, "data", "processed", "test_human_with_prompt.jsonl")
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        print("Please run data preparation script first")
        return
    
    test_samples = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
    
    # 随机选5个样本进行评估
    import random
    random.seed(42)
    selected_samples = random.sample(test_samples, min(5, len(test_samples)))
    
    print(f"Loaded {len(test_samples)} test samples, selected {len(selected_samples)} for evaluation")
    
    # 2. 加载模型
    base_model_path = "Qwen/Qwen3-7B"
    sft_adapter_path = os.path.join(project_dir, "models", "sft", "sft_20260105_1453")
    rl_adapter_path = os.path.join(project_dir, "models", "rl_detectgpt_pure_exp2_scaled", "grpo_detectgpt_pure_scaled_20260125")
    
    print("\n[Step 2] Loading SFT model...")
    sft_tokenizer, sft_model = load_model(base_model_path, sft_adapter_path)
    
    print("\n[Step 3] Loading RL model...")
    rl_tokenizer, rl_model = load_model(base_model_path, rl_adapter_path)
    
    # 3. 初始化DetectGPT
    print("\n[Step 4] Initializing DetectGPT detector...")
    detector = DetectGPTDetector.get_instance(
        model_name="Qwen/Qwen3-7B",
        num_perturbations=10,
        use_t5_perturbation=True
    )
    
    # 4. 对每个样本进行评估
    results = []
    
    for i, sample in enumerate(selected_samples, 1):
        print(f"\n{'='*80}")
        print(f"[Test Case {i}/{len(selected_samples)}]")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print('='*80)
        
        # 人类文本
        human_text = sample['text']
        human_len = len(human_text)
        
        # SFT生成
        print("\n[Generating with SFT...]")
        sft_text, sft_complete, sft_tokens = generate_text(
            sft_model, sft_tokenizer, sample['prompt'], max_tokens=300
        )
        print(f"  Tokens: {sft_tokens}/300, Complete: {sft_complete}, Chars: {len(sft_text)}")
        
        # RL生成
        print("[Generating with RL...]")
        rl_text, rl_complete, rl_tokens = generate_text(
            rl_model, rl_tokenizer, sample['prompt'], max_tokens=300
        )
        print(f"  Tokens: {rl_tokens}/300, Complete: {rl_complete}, Chars: {len(rl_text)}")
        
        # DetectGPT评估
        print("\n[Evaluating with DetectGPT...]")
        
        # 人类文本（截取前300 tokens对应的字符，保证公平对比）
        # 或者直接用完整的human text
        print("  [Human text]")
        human_score = detector.detect(human_text)
        human_likeness = 1.0 - human_score
        print(f"    DetectGPT: {human_score:.4f}, Human-likeness: {human_likeness:.4f}")
        
        print("  [SFT model]")
        sft_score = detector.detect(sft_text)
        sft_likeness = 1.0 - sft_score
        print(f"    DetectGPT: {sft_score:.4f}, Human-likeness: {sft_likeness:.4f}")
        
        print("  [RL model]")
        rl_score = detector.detect(rl_text)
        rl_likeness = 1.0 - rl_score
        print(f"    DetectGPT: {rl_score:.4f}, Human-likeness: {rl_likeness:.4f}")
        
        # 对比
        print("\n[Comparison]")
        print(f"  Human:  {human_likeness:.4f} (baseline)")
        print(f"  SFT:    {sft_likeness:.4f} ({(sft_likeness-human_likeness)*100:+.2f}%)")
        print(f"  RL:     {rl_likeness:.4f} ({(rl_likeness-human_likeness)*100:+.2f}%)")
        print(f"  RL vs SFT: {(rl_likeness-sft_likeness)*100:+.2f}%")
        
        results.append({
            "prompt": sample['prompt'],
            "human": {
                "text": human_text,
                "detectgpt": float(human_score),
                "human_likeness": float(human_likeness),
                "length": human_len
            },
            "sft": {
                "text": sft_text,
                "detectgpt": float(sft_score),
                "human_likeness": float(sft_likeness),
                "is_complete": sft_complete,
                "tokens": sft_tokens,
                "length": len(sft_text)
            },
            "rl": {
                "text": rl_text,
                "detectgpt": float(rl_score),
                "human_likeness": float(rl_likeness),
                "is_complete": rl_complete,
                "tokens": rl_tokens,
                "length": len(rl_text)
            }
        })
    
    # 保存结果
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comprehensive_evaluation.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("[Overall Statistics]")
    print('='*80)
    
    # 统计
    avg_human = sum(r['human']['human_likeness'] for r in results) / len(results)
    avg_sft = sum(r['sft']['human_likeness'] for r in results) / len(results)
    avg_rl = sum(r['rl']['human_likeness'] for r in results) / len(results)
    
    print(f"\nAverage Human-likeness:")
    print(f"  Human:  {avg_human:.4f} (baseline)")
    print(f"  SFT:    {avg_sft:.4f} (gap: {(avg_sft-avg_human)*100:+.2f}%)")
    print(f"  RL:     {avg_rl:.4f} (gap: {(avg_rl-avg_human)*100:+.2f}%)")
    print(f"  RL vs SFT: {(avg_rl-avg_sft)*100:+.2f}%")
    
    # 判断
    print(f"\n[Conclusion]")
    if avg_rl > avg_sft:
        improvement = ((avg_rl - avg_sft) / avg_sft) * 100
        print(f"✅ RL improved by {improvement:.2f}%")
        if avg_rl > avg_human:
            print(f"✅ RL even surpassed human baseline!")
        elif avg_rl < avg_human:
            gap = ((avg_human - avg_rl) / avg_human) * 100
            print(f"⚠️  But still {gap:.2f}% below human baseline")
    else:
        decline = ((avg_sft - avg_rl) / avg_sft) * 100
        print(f"❌ RL declined by {decline:.2f}%")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
