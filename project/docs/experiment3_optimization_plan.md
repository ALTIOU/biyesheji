# Experiment 3 优化计划

基于Experiment 2的失败教训，本文档提供下一个实验的具体优化方案。

---

## 快速决策：三个优化方案

### 🔥 方案A: 快速迭代 (推荐) ⭐⭐⭐⭐⭐

**适合**: 想快速验证改进效果，1-2天完成

**核心改动**:
1. 混合reward (DetectGPT 40% + Fluency 30% + Diversity 20% + Coherence 10%)
2. 更激进的reward scaling (scale_factor=20)
3. 优化超参数 (lr=1e-5, patience=50)

**预期**: 60-70%成功概率，效果提升5-10%

**实施**: 见下方详细配置

---

### 🚀 方案B: 稳扎稳打 ⭐⭐⭐⭐

**适合**: 想要更可靠的结果，3-5天完成

**核心改动**:
1. 使用不同的评估模型 (Mistral-7B或LLaMA-2)
2. 混合reward + 对比学习
3. 收集100对人工标注数据

**预期**: 70-80%成功概率，效果提升10-20%

**实施**: 需要额外准备工作（安装新模型、标注数据）

---

### 💎 方案C: 大力出奇迹 ⭐⭐⭐

**适合**: 愿意投入更多资源，1周完成

**核心改动**:
1. 升级到Qwen3-4B或7B模型
2. 租用云GPU (32GB+ VRAM)
3. 完整的RLHF pipeline

**预期**: 80-90%成功概率，效果提升20-30%+

**实施**: 需要额外预算（云GPU约$50-100）

---

## 方案A详细实施（推荐）

### Step 1: 改进Reward Function

**文件**: `src/rl/reward_functions.py`

```python
def mixed_reward_with_info(
    text: str,
    prompt: str = "",
    **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    混合奖励函数：DetectGPT + Fluency + Diversity + Coherence
    
    权重分配：
    - DetectGPT: 40% (仍然是主要信号)
    - Fluency: 30% (通过perplexity衡量)
    - Diversity: 20% (unique n-grams)
    - Coherence: 10% (与prompt的语义相关性)
    """
    info = {}
    
    # 1. DetectGPT component (40%)
    detectgpt_detector = DetectGPTDetector.get_instance(
        model_name="Qwen/Qwen3-7B",
        num_perturbations=15,  # 增加到15次
        use_t5_perturbation=True
    )
    detection_score = detectgpt_detector.detect(text)
    detectgpt_reward = 1.0 - detection_score
    info["detectgpt_score"] = float(detection_score)
    info["detectgpt_reward"] = float(detectgpt_reward)
    
    # 2. Fluency component (30%)
    # 使用perplexity的倒数作为fluency指标
    try:
        inputs = detectgpt_detector.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(detectgpt_detector.device)
        
        with torch.no_grad():
            outputs = detectgpt_detector.model(**inputs, labels=inputs["input_ids"])
            perplexity = torch.exp(outputs.loss).item()
        
        # Normalize perplexity to [0, 1]
        # Good text: ppl ~10-30, bad text: ppl >100
        fluency_reward = max(0.0, min(1.0, 1.0 - (perplexity - 10) / 90))
        info["perplexity"] = float(perplexity)
        info["fluency_reward"] = float(fluency_reward)
    except Exception as e:
        fluency_reward = 0.5  # default if error
        info["fluency_error"] = str(e)
    
    # 3. Diversity component (20%)
    # 计算unique n-grams的比例
    words = text.lower().split()
    if len(words) < 10:
        diversity_reward = 0.5
    else:
        # Unique bigrams / total bigrams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        diversity_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 0.5
        diversity_reward = min(1.0, diversity_ratio * 1.2)  # slight boost
    
    info["diversity_ratio"] = float(diversity_ratio) if 'diversity_ratio' in locals() else 0.5
    info["diversity_reward"] = float(diversity_reward)
    
    # 4. Coherence component (10%)
    # 简单实现：检查文本长度是否合理，避免过短或重复
    text_length = len(text)
    has_repetition = check_repetition(text)
    
    if text_length < 50:
        coherence_reward = 0.3  # too short
    elif text_length > 2000:
        coherence_reward = 0.7  # might be too long
    elif has_repetition:
        coherence_reward = 0.4  # repetitive
    else:
        coherence_reward = 0.9  # good
    
    info["text_length"] = text_length
    info["has_repetition"] = has_repetition
    info["coherence_reward"] = float(coherence_reward)
    
    # 5. 混合final reward
    final_reward = (
        0.40 * detectgpt_reward +
        0.30 * fluency_reward +
        0.20 * diversity_reward +
        0.10 * coherence_reward
    )
    
    # 6. Reward Scaling (更激进)
    baseline = 0.48  # 基于Exp2的实际测试调整
    scale_factor = 20.0  # 从5提升到20
    scaled_reward = 0.5 + (final_reward - baseline) * scale_factor
    scaled_reward = max(0.0, min(1.0, scaled_reward))
    
    info["final_reward_raw"] = float(final_reward)
    info["final_reward_scaled"] = float(scaled_reward)
    info["component_weights"] = {
        "detectgpt": 0.40,
        "fluency": 0.30,
        "diversity": 0.20,
        "coherence": 0.10
    }
    
    return float(scaled_reward), info


def check_repetition(text: str, max_repeat: int = 3) -> bool:
    """检查是否有连续重复的短语"""
    words = text.split()
    for length in [3, 4, 5]:  # check 3/4/5-grams
        for i in range(len(words) - length * max_repeat):
            phrase = " ".join(words[i:i+length])
            rest = " ".join(words[i+length:])
            if rest.count(phrase) >= max_repeat:
                return True
    return False


# 注册到reward registry
REWARD_FUNCTIONS = {
    ...
    "mixed_reward": mixed_reward_with_info,
}
```

### Step 2: 更新训练脚本配置

**文件**: `start_sh/run_grpo_exp3.sh`

```bash
#!/bin/bash

# =====================================================
# Experiment 3: Mixed Reward with Enhanced Scaling
# =====================================================

# Load environment
set -a
source biyesheji/project/.env.local
set +a

# Export experiment-specific settings
export USE_T5_PERTURBATION=1
export REWARD_BASELINE=0.48
export REWARD_SCALE_FACTOR=20  # Much more aggressive

# Experiment configuration
BASE_MODEL="Qwen/Qwen3-7B"
SFT_MODEL="biyesheji/project/models/sft/sft_20260105_1453"
OUTPUT_DIR="biyesheji/project/models/rl_mixed_reward_exp3"
DATA_PATH="biyesheji/project/data/processed/rl_prompts.jsonl"
RUN_NAME="grpo_mixed_reward_exp3_$(date +%Y%m%d_%H%M)"

# Hyperparameters (optimized)
LR=1e-5              # More conservative than 2e-5
WARMUP=20            # Longer warmup
EARLY_STOP=50        # More patient
GROUP_SIZE=12        # Slightly larger if VRAM allows
MAX_TOKENS=200       # Keep same
MAX_PROMPTS=100      # Full dataset

# Log configuration
echo "========================================"
echo "Experiment 3: Mixed Reward Function"
echo "========================================"
echo "Base Model: $BASE_MODEL"
echo "SFT Model: $SFT_MODEL"
echo "Reward: mixed_reward (DetectGPT 40% + Fluency 30% + Diversity 20% + Coherence 10%)"
echo "Learning Rate: $LR"
echo "Group Size: $GROUP_SIZE"
echo "Max Tokens: $MAX_TOKENS"
echo "Max Prompts: $MAX_PROMPTS"
echo "Warmup Steps: $WARMUP"
echo "Early Stopping Patience: $EARLY_STOP"
echo "Reward Scaling: baseline=$REWARD_BASELINE, factor=$REWARD_SCALE_FACTOR"
echo "========================================"
echo ""

# Run training
conda run -n biyeshejihuanjing python biyesheji/project/src/rl/train_grpo.py \
    --model_name "$BASE_MODEL" \
    --sft_model_path "$SFT_MODEL" \
    --dataset_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --reward_fn "mixed_reward" \
    --learning_rate $LR \
    --num_epochs 1 \
    --group_size $GROUP_SIZE \
    --max_new_tokens $MAX_TOKENS \
    --max_prompts $MAX_PROMPTS \
    --warmup_steps $WARMUP \
    --early_stopping_patience $EARLY_STOP \
    --save_steps 10 \
    --logging_steps 1 \
    --use_wandb

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"
```

### Step 3: 修改超参数

**关键改动**:

```yaml
Before (Exp2):
  learning_rate: 2e-5
  warmup_steps: 15
  patience: 30
  scale_factor: 5
  num_perturbations: 10

After (Exp3):
  learning_rate: 1e-5        # ↓ 更保守，避免震荡
  warmup_steps: 20           # ↑ 更长预热
  patience: 50               # ↑ 更宽容
  scale_factor: 20           # ↑ 更强信号
  num_perturbations: 15      # ↑ 更稳定
```

### Step 4: 启动训练

```bash
cd biyesheji/project
bash start_sh/run_grpo_exp3.sh
```

**预计时间**: 3-4小时（100 prompts）

### Step 5: 评估

使用改进的评估脚本（fix编码问题）：

```bash
python scripts/comprehensive_evaluation_exp3.py
```

---

## 预期结果分析

### 成功指标

```yaml
Minimum Success:
  RL vs SFT: +2% human-likeness
  
Good Success:
  RL vs SFT: +5% human-likeness
  Training curve stable, loss decreases
  
Excellent Success:
  RL vs SFT: +10% human-likeness
  All test cases improved
```

### 如果仍然失败

**可能原因**:
1. Reward signal仍然不够强 → 尝试scale_factor=50
2. 模型容量不足 → 升级到Qwen3-4B（方案C）
3. 需要真实人类反馈 → 转向RLHF（方案B）

---

## 方案B详细实施（备选）

### 核心思路：使用不同的评估模型

**问题**: Exp2中生成模型==评估模型，可能有bias

**解决**: 使用**不同家族**的模型作为evaluator

### 实施步骤

#### 1. 下载并配置Mistral-7B

```bash
# Install if needed
pip install accelerate

# Download model (one-time)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'mistralai/Mistral-7B-v0.1'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Download complete!')
"
```

#### 2. 修改DetectGPT评估器

```python
# In reward_functions.py
class DetectGPTDetector:
    _instances = {}
    
    @classmethod
    def get_instance(
        cls, 
        model_name: str = "mistralai/Mistral-7B-v0.1",  # Changed!
        num_perturbations: int = 10,
        use_t5_perturbation: bool = False
    ):
        # ... rest same
```

#### 3. 配置VRAM管理

Mistral-7B需要更多VRAM：

```python
# Load with 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**VRAM估算**:
```
Qwen3-1.7B (训练): ~8GB
Mistral-7B (8-bit评估): ~7GB
T5-small (扰动): ~1GB
Overhead: ~1GB
Total: ~17GB (需要20GB+ VRAM，或需要offload部分到CPU)
```

**替代方案**: 如果VRAM不够，使用LLaMA-2-7B + 更激进的量化

---

## 方案C详细实施（终极方案）

### 核心思路：全面升级

#### 1. 租用云GPU

**推荐平台**:
```
Option 1: AutoDL (国内)
  - Tesla V100 32GB: ¥3/hour
  - RTX 4090 24GB: ¥2.5/hour
  
Option 2: RunPod (国际)
  - A100 40GB: $1.89/hour
  - RTX A6000 48GB: $0.79/hour

Option 3: Vast.ai (最便宜)
  - RTX 3090 24GB: $0.2-0.5/hour
```

**预算估算**:
- 训练时间: 4-6小时
- 总成本: $5-15

#### 2. 升级模型到Qwen3-4B

```yaml
Model: Qwen/Qwen3-4B
VRAM (FP16): ~8GB (base)
VRAM (LoRA training): ~12-14GB
Total with DetectGPT: ~18-20GB (需要24GB+ GPU)
```

#### 3. 完整RLHF Pipeline

**Phase 1**: 收集人类偏好数据
```python
# Generate comparison pairs
for prompt in sample_prompts[:100]:
    sft_text = sft_model.generate(prompt)
    rl_text = rl_model.generate(prompt)
    
    # Manual annotation
    print(f"Prompt: {prompt}")
    print(f"A: {sft_text}")
    print(f"B: {rl_text}")
    choice = input("Which is more human-like? (A/B/Tie): ")
    
    labels.append({
        "prompt": prompt,
        "text_a": sft_text,
        "text_b": rl_text,
        "choice": choice
    })
```

**Phase 2**: 训练Reward Model
```python
# Train a preference model
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1
)

# Train on labeled pairs
# ... (standard binary classification)
```

**Phase 3**: 使用Reward Model进行RL
```python
def rlhf_reward(text, prompt):
    score = reward_model(text, prompt)
    return score.item()
```

---

## 风险与应对

### Risk 1: 混合reward权重不当

**应对**: 
- 先用小数据集（10-20 prompts）快速测试
- 调整权重，找到最优配比
- 记录每个component的贡献度

### Risk 2: VRAM不足（方案B/C）

**应对**:
- 使用8-bit或4-bit量化
- Gradient checkpointing
- 减少batch size
- CPU offloading

### Risk 3: 训练时间过长

**应对**:
- 减少max_prompts (100 → 50)
- 减少num_perturbations (15 → 10)
- 使用更快的GPU

---

## Quick Start指南

### 如果你现在就想开始

**最快路径** (30分钟内启动):

```bash
# 1. 复制并修改reward function
cd biyesheji/project
cp src/rl/reward_functions.py src/rl/reward_functions.py.backup

# 2. 添加mixed_reward函数（见上方代码）
# ... edit reward_functions.py ...

# 3. 创建训练脚本
cp start_sh/run_grpo_exp2.sh start_sh/run_grpo_exp3.sh
# ... edit run_grpo_exp3.sh ...

# 4. 启动训练
bash start_sh/run_grpo_exp3.sh

# 5. 监控WandB
# Open: https://wandb.ai/your-username/your-project
```

---

## 总结建议

### 我的推荐

**如果时间有限（毕设deadline）**: 
→ **方案A** (混合reward + 优化超参数)

**如果想要更好效果**: 
→ **方案B** (不同评估模型) + **收集50-100对人工标注**

**如果有预算且追求完美**: 
→ **方案C** (大模型 + 云GPU + 完整RLHF)

### 实施时间表

```
Day 1:
  - 修改reward function (2-3小时)
  - 测试代码运行（1小时）
  - 启动训练（4-6小时，可过夜）

Day 2:
  - 分析结果（1小时）
  - 如果失败，调整参数重试（4-6小时）
  
Day 3:
  - 最终评估和文档（2-3小时）
```

---

**Good Luck! 🚀**

如有问题随时问我，我会帮你调试！
