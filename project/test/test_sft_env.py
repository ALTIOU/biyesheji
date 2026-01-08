#!/usr/bin/env python3
"""
测试 SFT 训练环境是否配置正确
"""

import os
import sys
import json
from pathlib import Path

def test_imports():
    """测试必要的包是否能导入"""
    print("🔧 测试包导入...")
    try:
        import torch
        print(f"✅ torch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")

        import transformers
        print(f"✅ transformers 版本: {transformers.__version__}")

        import peft
        print(f"✅ peft 版本: {peft.__version__}")

        import accelerate
        print(f"✅ accelerate 版本: {accelerate.__version__}")

        import datasets
        print(f"✅ datasets 版本: {datasets.__version__}")

        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n📁 测试数据加载...")
    data_path = "../data/raw/dataset_prepared/sft_data.jsonl"

    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                item = json.loads(line)
                if count < 2:  # 只显示前2个样本
                    print(f"   示例 {count+1}: {item['instruction'][:50]}...")
                count += 1
                if count >= 10:  # 只读取前10个
                    break

        print(f"✅ 数据加载成功，共 {count} 条记录")
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_tokenizer():
    """测试 tokenizer 加载"""
    print("\n🔤 测试 tokenizer...")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
        print("✅ Tokenizer 加载成功")

        # 测试编码
        test_text = "你好，这是测试文本。"
        tokens = tokenizer.encode(test_text)
        print(f"✅ 编码测试: '{test_text}' -> {len(tokens)} tokens")

        return True
    except Exception as e:
        print(f"❌ Tokenizer 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始 SFT 训练环境测试\n")

    results = []
    results.append(("包导入", test_imports()))
    results.append(("数据加载", test_data_loading()))
    results.append(("Tokenizer", test_tokenizer()))

    print("\n" + "="*50)
    print("📊 测试结果总结:")

    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n🎉 所有测试通过！环境配置正确，可以开始训练。")
        print("\n运行训练命令:")
        print("cd /Users/altiou/code_learn/毕业设计")
        print("conda activate biyesheji")
        print("python project/src/sft/train_sft_lora.py")
    else:
        print("\n⚠️  部分测试失败，请检查环境配置。")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
