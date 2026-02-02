#!/bin/bash
# ==============================================================================
# 实验2: DetectGPT Pure Reward (Qwen3-1.7B评估) - Reward Scaling优化版
# ==============================================================================
# 
# 主要配置:
# 1. 评估模型: Qwen3-1.7B (与训练模型完全匹配)
# 2. 奖励函数: detectgpt_pure (纯DetectGPT，无混合因子)
# 3. **Reward Scaling**: 5倍放大差异 (解决reward变化太小问题)
# 4. 扰动次数: 10 (提升准确性，原论文用100)
# 5. 扰动方法: T5-small mask-filling (原始论文方法)
# 6. **学习率: 2e-5** (从5e-6增大4倍，加快收敛)
# 7. Warmup: 前15步线性增长学习率 (从10改为15)
# 8. Early stopping: 连续30步无提升则停止 (从20改为30，更宽容)
# 9. **Batch设置: group_size=10, max_new_tokens=200** (平衡显存和性能)
# 10. Prompt数量: 100 (快速验证)
#
# 关键改进:
# - Reward Scaling: 将0.44-0.50的微小变化放大到0.35-0.65
# - Learning Rate: 增大到2e-5，配合放大的reward更快学习
# - 更宽容的early stopping: 给模型更多探索时间
#
# 预估时间: 6-8小时
# ==============================================================================

# 加载环境变量
source biyesheji/project/.env.local

# 激活conda环境
conda activate biyeshejihuanjing

# 启用 T5 扰动
export USE_T5_PERTURBATION=1

# 运行训练
python biyesheji/project/src/rl/train_grpo.py \
  --base_model Qwen/Qwen3-7B \
  --sft_adapter_path biyesheji/project/models/sft/sft_20260105_1453 \
  --rl_data_path biyesheji/project/data/processed/rl_prompts.jsonl \
  --output_dir biyesheji/project/models/rl_detectgpt_pure_exp2_scaled \
  --reward_name detectgpt_pure \
  --max_prompts 100 \
  --group_size 10 \
  --max_new_tokens 200 \
  --learning_rate 2e-5 \
  --warmup_steps 15 \
  --early_stopping_patience 30 \
  --report_to wandb \
  --wandb_project graduation_grpo \
  --run_name grpo_detectgpt_pure_scaled_$(date +%Y%m%d)

echo ""
echo "[Success] Training completed!"
echo "Output directory: biyesheji/project/models/rl_detectgpt_pure_exp2_scaled"
echo ""
echo "Configuration Summary:"
echo "  - Reward Scaling: 5x amplification (0.44-0.50 -> 0.35-0.65)"
echo "  - Learning Rate: 2e-5 (4x larger than before)"
echo "  - Group Size: 10 (balanced for 16GB VRAM)"
echo "  - Max Tokens: 200 (reduced from 256)"
echo "  - Perturbations: 10 (T5-small method)"
echo "  - Warmup: 15 steps"
echo "  - Early Stopping: 30 steps patience"
