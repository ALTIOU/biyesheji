"""
RL模型评估脚本
评估指标：BLEU, ROUGE, Perplexity, DetectGPT等
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import numpy as np


def load_model_and_tokenizer(base_model, adapter_path, device):
    """加载模型和tokenizer"""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    if adapter_path:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    model.eval()
    return model, tokenizer


def load_test_data(test_file):
    """加载测试数据"""
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def generate_responses(model, tokenizer, prompts, max_new_tokens=320, device="cuda"):
    """批量生成回复"""
    responses = []
    
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.95,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        responses.append(response)
    
    return responses


def compute_perplexity(model, tokenizer, texts, device="cuda"):
    """计算困惑度"""
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(texts, desc="Computing perplexity"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        total_loss += loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def compute_bleu(references, hypotheses):
    """计算BLEU分数"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        
        # 下载必要的数据（首次运行）
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        smoothie = SmoothingFunction().method4
        scores = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = ref.split()
            hyp_tokens = hyp.split()
            
            if len(hyp_tokens) == 0:
                scores.append(0.0)
            else:
                score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
                scores.append(score)
        
        return np.mean(scores)
    except ImportError:
        print("[Warning] NLTK not installed. Skipping BLEU calculation.")
        return None


def compute_rouge(references, hypotheses):
    """计算ROUGE分数"""
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
        }
    except ImportError:
        print("[Warning] rouge-score not installed. Skipping ROUGE calculation.")
        return None


def compute_detectgpt_score(texts):
    """计算DetectGPT分数"""
    try:
        # 导入你的奖励函数
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "rl"))
        from reward_functions import detectgpt_reward
        
        scores = []
        for text in tqdm(texts, desc="Computing DetectGPT scores"):
            score = detectgpt_reward(text)
            scores.append(score)
        
        return np.mean(scores)
    except Exception as e:
        print(f"[Warning] DetectGPT calculation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="评估RL训练后的模型")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-7B")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter路径")
    parser.add_argument("--test_file", type=str, default="biyesheji/project/data/processed/test_human.jsonl")
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=100, help="评估样本数量，0表示全部")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path, args.device)
    
    # 加载测试数据
    print(f"\nLoading test data from: {args.test_file}")
    test_data = load_test_data(args.test_file)
    
    if args.num_samples > 0:
        test_data = test_data[:args.num_samples]
    
    print(f"Total test samples: {len(test_data)}")
    
    # 提取prompts和references（如果有）
    prompts = [item.get("prompt", item.get("instruction", "")) for item in test_data]
    references = [item.get("output", item.get("response", "")) for item in test_data]
    
    # 生成回复
    print("\n" + "="*60)
    print("Generating responses...")
    print("="*60)
    responses = generate_responses(model, tokenizer, prompts, args.max_new_tokens, args.device)
    
    # 计算评估指标
    results = {
        "model": args.base_model,
        "adapter_path": args.adapter_path,
        "num_samples": len(test_data),
    }
    
    print("\n" + "="*60)
    print("Computing evaluation metrics...")
    print("="*60)
    
    # 1. 困惑度
    print("\n[1/4] Computing Perplexity...")
    perplexity = compute_perplexity(model, tokenizer, responses, args.device)
    results["perplexity"] = float(perplexity)
    print(f"Perplexity: {perplexity:.2f}")
    
    # 2. BLEU (需要参考答案)
    if references and all(ref for ref in references):
        print("\n[2/4] Computing BLEU...")
        bleu = compute_bleu(references, responses)
        if bleu is not None:
            results["bleu"] = float(bleu)
            print(f"BLEU: {bleu:.4f}")
    else:
        print("\n[2/4] Skipping BLEU (no references)")
        results["bleu"] = None
    
    # 3. ROUGE (需要参考答案)
    if references and all(ref for ref in references):
        print("\n[3/4] Computing ROUGE...")
        rouge = compute_rouge(references, responses)
        if rouge is not None:
            results["rouge"] = rouge
            print(f"ROUGE-1: {rouge['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge['rougeL']:.4f}")
    else:
        print("\n[3/4] Skipping ROUGE (no references)")
        results["rouge"] = None
    
    # 4. DetectGPT分数
    print("\n[4/4] Computing DetectGPT score...")
    detectgpt_score = compute_detectgpt_score(responses)
    if detectgpt_score is not None:
        results["detectgpt_score"] = float(detectgpt_score)
        print(f"DetectGPT Score: {detectgpt_score:.4f}")
    
    # 保存结果
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")
    
    # 保存生成样例
    samples_file = output_path.parent / f"{output_path.stem}_samples.jsonl"
    with open(samples_file, 'w', encoding='utf-8') as f:
        for prompt, response in zip(prompts[:10], responses[:10]):  # 保存前10个样例
            f.write(json.dumps({
                "prompt": prompt,
                "response": response
            }, ensure_ascii=False) + "\n")
    
    print(f"Sample responses saved to: {samples_file}")
    
    # 打印总结
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Samples: {len(test_data)}")
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.2f}" if results.get('perplexity') else "Perplexity: N/A")
    print(f"BLEU: {results.get('bleu', 'N/A'):.4f}" if results.get('bleu') else "BLEU: N/A")
    if results.get('rouge'):
        print(f"ROUGE-1: {results['rouge']['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge']['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
    print(f"DetectGPT: {results.get('detectgpt_score', 'N/A'):.4f}" if results.get('detectgpt_score') else "DetectGPT: N/A")
    print("="*60)


if __name__ == "__main__":
    main()
