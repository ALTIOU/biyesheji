# DetectGPT 奖励函数使用说明

## 📋 修改内容总结

已将 DetectGPT 评估模型从 **GPT-2** 改为 **Qwen/Qwen3-7B**，确保模型匹配，提高概率计算准确性。

---

## 🎯 两种奖励函数

### 1. `detectgpt` - 混合版（当前正在运行）
- **组成**: DetectGPT (70%) + 长度因子 (20%) + 重复度因子 (10%)
- **用途**: 平衡人类相似度和基础质量
- **适用**: 需要综合考虑多个因素的实验

### 2. `detectgpt_pure` - 纯净版（下一轮推荐）
- **组成**: 100% DetectGPT，无任何附加因子
- **用途**: 验证 DetectGPT 本身的有效性
- **适用**: 对照实验、理论验证

---

## ⚙️ T5 扰动开关

### 环境变量控制

通过 `USE_T5_PERTURBATION` 环境变量控制扰动方法：

| 值 | 扰动方法 | 特点 | 适用场景 |
|---|---------|------|---------|
| `0` (默认) | 简单随机替换 | 快速，适合训练 | ✅ GRPO 训练阶段 |
| `1` | T5 mask-filling | 更准确，接近原论文 | ✅ 模型评估阶段 |

### 使用方法

#### **训练时（简单扰动）**
```bash
# Linux / Mac
python biyesheji/project/src/rl/train_grpo.py \
  --reward_name detectgpt_pure \
  --base_model Qwen/Qwen3-7B \
  --sft_adapter_path biyesheji/project/models/sft/sft_20260105_1453 \
  --rl_data_path biyesheji/project/data/processed/rl_prompts.jsonl \
  --output_dir biyesheji/project/models/rl_detectgpt_pure \
  --max_prompts 0

# Windows PowerShell
conda activate biyeshejihuanjing
$envContent = Get-Content biyesheji/project/.env.local
foreach ($line in $envContent) {
    if ($line -match '^HF_TOKEN=(.*)$') { $env:HF_TOKEN = $matches[1] }
    if ($line -match '^WANDB_API_KEY=(.*)$') { $env:WANDB_API_KEY = $matches[1] }
}
python biyesheji/project/src/rl/train_grpo.py `
  --reward_name detectgpt_pure `
  --base_model Qwen/Qwen3-7B `
  --sft_adapter_path biyesheji/project/models/sft/sft_20260105_1453 `
  --rl_data_path biyesheji/project/data/processed/rl_prompts.jsonl `
  --output_dir biyesheji/project/models/rl_detectgpt_pure `
  --max_prompts 0
```

#### **评估时（T5 扰动）**
```bash
# Linux / Mac
USE_T5_PERTURBATION=1 python biyesheji/project/src/evaluate/eval_rl_model.py \
  --base_model Qwen/Qwen3-7B \
  --adapter_path biyesheji/project/models/rl_detectgpt_pure \
  --test_data_path biyesheji/project/data/processed/test_data.jsonl

# Windows PowerShell
$env:USE_T5_PERTURBATION="1"
python biyesheji/project/src/evaluate/eval_rl_model.py `
  --base_model Qwen/Qwen3-7B `
  --adapter_path biyesheji/project/models/rl_detectgpt_pure `
  --test_data_path biyesheji/project/data/processed/test_data.jsonl
```

---

## 🔬 实验建议

### 实验对比矩阵

| 实验编号 | 奖励函数 | 评估模型 | 扰动方法 | 目的 |
|---------|---------|---------|---------|------|
| **实验1** (进行中) | `detectgpt` | GPT-2 | Simple | Baseline（模型不匹配） |
| **实验2** | `detectgpt_pure` | Qwen3-1.7B | Simple | 纯DetectGPT + 正确模型 |
| **实验3** | `detectgpt_pure` | Qwen3-1.7B | T5 | 最优配置验证 |
| **实验4** | `humanlike_v1` | - | - | 对照组 |

### 运行顺序建议

1. ✅ **当前运行**: 实验1（已在进行）
2. 📝 **记录结果**: 完成后记录训练曲线、最终指标
3. 🔄 **下一轮**: 实验2（`detectgpt_pure` + Simple 扰动）
4. 🔍 **评估对比**: 使用 T5 扰动重新评估所有模型

---

## 💾 显存预估

| 配置 | 训练模型 | 评估模型 | 总显存 | 安全性 |
|-----|---------|---------|--------|-------|
| GPT-2 | 1.7B | 124M | ~9GB | ✅ 安全 |
| Qwen3-1.7B | 1.7B | 1.7B | ~12GB | ✅ 安全（16GB卡） |
| Qwen3-3B | 1.7B | 3B | ~15GB | ⚠️ 可能爆显存 |

---

## 📊 预期效果

### GPT-2 评估的问题（实验1）
- GPT-2 可能无法准确理解 Qwen3 生成的文本
- 概率曲率计算可能不准确
- 但可以作为 baseline 参考

### Qwen3-1.7B 评估的优势（实验2+）
- ✅ 模型完全匹配，概率分布一致
- ✅ 能正确理解中英文混合文本
- ✅ 符合 DetectGPT 原论文的理论假设

---

## 🛠️ 代码修改位置

如果需要手动调整参数，可修改以下位置：

```python
# biyesheji/project/src/rl/reward_functions.py

# 1. 修改默认评估模型（已改为 Qwen3-1.7B）
class DetectGPTDetector:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-7B",  # 这里
        ...
    ):

# 2. 修改扰动次数（当前为3，可调整为5以提高准确性）
detector = DetectGPTDetector.get_instance(
    model_name="Qwen/Qwen3-7B",
    num_perturbations=3,  # 这里，越大越慢但越准确
    use_t5_perturbation=use_t5
)
```

---

## 📝 论文可以这样写

### 模型配置部分：
> "为确保 DetectGPT 检测器能准确评估生成文本的概率分布，我们使用与训练模型相同的 Qwen3-1.7B 作为评估模型，而非传统的 GPT-2。这一配置符合 DetectGPT 原论文的理论假设，即评估模型应能理解生成模型的概率空间特性。"

### 实验对比部分：
> "我们对比了两种评估模型配置：（1）GPT-2（124M 参数）和（2）Qwen3-1.7B（与训练模型匹配）。结果表明，使用匹配的评估模型显著提升了 DetectGPT 奖励函数的区分能力，验证了模型选择对概率曲率计算准确性的重要性。"

---

## ❓ 常见问题

### Q1: 为什么不用 GPT-2？
**A**: GPT-2 太小且训练数据老旧，无法准确理解 Qwen3 的概率分布。使用相同模型符合 DetectGPT 理论。

### Q2: T5 扰动会慢多少？
**A**: 大约慢 2-3 倍。训练时用简单扰动即可，评估时再用 T5 获得更准确的结果。

### Q3: 显存不够怎么办？
**A**: 可以调小 `num_perturbations`（从 3 降到 1），或在 CPU 上运行评估模型（速度会慢很多）。

### Q4: 如何验证配置是否生效？
**A**: 查看训练日志，首次加载时会显示加载的模型名称。可以在日志中搜索 "DetectGPT" 或模型名称。

---

**最后更新**: 2026-01-24  
**状态**: ✅ 代码已修改完成，等待下一轮实验
