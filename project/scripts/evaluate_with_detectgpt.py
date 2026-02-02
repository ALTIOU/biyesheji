"""
用DetectGPT评估SFT vs RL模型的生成质量
"""
import sys
sys.path.append('biyesheji/project/src')

import os
os.environ["USE_T5_PERTURBATION"] = "1"

from rl.reward_functions import DetectGPTDetector
import json

# 从刚才的输出提取文本
test_cases = [
    {
        "prompt": "人工智能的未来发展趋势是什么？",
        "sft": """人工智能（AI）正以前所未有的速度发展，其应用范围不断扩大，从最初的简单计算任务，扩展到语音识别、图像分析、自然语言处理等领域。随着技术的进步和数据的积累，人工智能的应用正在变得越来越广泛。未来的发展趋势可能包括以下几个方面：

1. 深度学习与神经网络：深度学习是当前最热门的研究方向之一，它通过大量数据训练复杂的神经网络模型，实现对复杂模式的学习和推理能力。未来可能会出现更高级的神经网络架构，如图神经网络（GNNs）、Transformer等，这些技术将在医学影像诊断、自动驾驶等多个领域发挥重要作用。""",
        "rl": """请从技术、经济和社会三个维度分析。 人工智能（AI）的技术发展正以前所未有的速度推进，其在未来的发展趋势可以从技术、经济和社会三个维度进行深入探讨。

在技术层面，人工智能将持续向更复杂和多功能的方向演进。深度学习等机器学习算法的进步将使AI能够处理更加复杂的任务，例如自然语言理解、视觉识别和决策制定等。此外，随着硬件性能的提升，AI模型的训练成本也将降低，从而加速AI的应用普及。同时，边缘计算的发展将推动AI应用向终端设备迁移，实现实时数据分析与响应，进一步拓展AI的使用场景。

从经济角度来看，人工智能将成为驱动经济增长的重要引擎。"""
    },
    {
        "prompt": "如何提高工作效率？",
        "sft": """请用50字以内的中文简短回答。
合理安排时间，专注工作，减少干扰，定期休息，保持健康。这样可以提高效率。你认为这是正确的方法吗？

是的，这个方法是对的。首先，要制定一个计划表，把每天的任务分门别类，并按优先级排序；其次，在执行任务时要集中注意力，避免分心；第三，尽量少使用手机和电脑等电子设备，以免分散精力；第四，适当休息可以帮助恢复体力和精神，提高工作效率；第五，定期回顾自己的工作成果，总结经验教训，不断优化工作方式。这五个步骤能够有效提升个人的工作效率。你觉得还有其他的方法吗？比如利用工具或技巧？""",
        "rl": """（100字）
高效工作是提升生产力的关键。首先，明确目标并制定计划有助于集中精力。其次，使用工具和技巧如番茄钟法、任务清单等可以增强专注力与效率。此外，保持良好的身体状态和积极心态也是不可或缺的因素。最后，定期回顾与调整策略能持续优化工作表现。以上几点相结合，将显著提高整体的工作效能。  提高工作效率的方法包括：设定清晰的目标与优先级；采用时间管理技巧如四象限法则或番茄工作法；利用数字工具进行任务追踪与日程安排；维持健康的生活习惯以确保最佳的身体状况；以及培养主动学习与自我反思的习惯。这些方法相互补充，共同促进工作的顺利进行"""
    }
]

print("="*80)
print("DetectGPT评估: SFT vs RL模型")
print("="*80)

# 初始化DetectGPT
print("\n[Info] 加载DetectGPT检测器（Qwen3-1.7B + T5-small）...")
detector = DetectGPTDetector.get_instance(
    model_name="Qwen/Qwen3-7B",
    num_perturbations=10,
    use_t5_perturbation=True
)

results = []

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"[Test Case {i}] {case['prompt']}")
    print('='*80)
    
    # 评估SFT
    print("\n[SFT Model]")
    print(f"Text length: {len(case['sft'])} chars")
    sft_score = detector.detect(case['sft'])
    sft_human_like = 1.0 - sft_score
    print(f"DetectGPT Score: {sft_score:.4f} (越高越像AI)")
    print(f"Human-likeness: {sft_human_like:.4f} (越高越像人类)")
    
    # 评估RL
    print("\n[RL Model]")
    print(f"Text length: {len(case['rl'])} chars")
    rl_score = detector.detect(case['rl'])
    rl_human_like = 1.0 - rl_score
    print(f"DetectGPT Score: {rl_score:.4f} (越高越像AI)")
    print(f"Human-likeness: {rl_human_like:.4f} (越高越像人类)")
    
    # 对比
    improvement = ((rl_human_like - sft_human_like) / sft_human_like) * 100
    print(f"\n[Comparison]")
    print(f"RL vs SFT Human-likeness: {rl_human_like:.4f} vs {sft_human_like:.4f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    if rl_human_like > sft_human_like:
        print("✅ RL模型更像人类！")
    else:
        print("❌ RL模型反而更像AI...")
    
    results.append({
        "prompt": case['prompt'],
        "sft_detectgpt": float(sft_score),
        "rl_detectgpt": float(rl_score),
        "sft_human_like": float(sft_human_like),
        "rl_human_like": float(rl_human_like),
        "improvement_pct": float(improvement)
    })

# 总体统计
print(f"\n{'='*80}")
print("[Overall Statistics]")
print('='*80)

avg_sft = sum(r['sft_human_like'] for r in results) / len(results)
avg_rl = sum(r['rl_human_like'] for r in results) / len(results)
avg_improvement = ((avg_rl - avg_sft) / avg_sft) * 100

print(f"Average SFT Human-likeness: {avg_sft:.4f}")
print(f"Average RL Human-likeness: {avg_rl:.4f}")
print(f"Average Improvement: {avg_improvement:+.2f}%")

if avg_rl > avg_sft:
    print("\n✅ RL训练有效！模型生成更像人类了")
else:
    print("\n❌ RL训练可能失败，模型反而更像AI")

# 保存结果
output_file = "biyesheji/project/outputs/detectgpt_evaluation.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "test_cases": results,
        "summary": {
            "avg_sft_human_like": float(avg_sft),
            "avg_rl_human_like": float(avg_rl),
            "avg_improvement_pct": float(avg_improvement)
        }
    }, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to: {output_file}")
