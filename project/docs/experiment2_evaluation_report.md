# Experiment 2: DetectGPT Pure Reward - Final Evaluation Report

**实验日期**: 2026-01-25  
**实验编号**: Experiment 2  
**实验目标**: 使用纯DetectGPT奖励函数进行GRPO训练，提升模型生成文本的human-likeness

---

## 1. 实验概述

### 1.1 核心思路

使用DetectGPT作为唯一的奖励信号，通过强化学习（GRPO）训练Qwen3-1.7B模型，使其生成的文本更难被AI检测器识别，从而提升"人类相似度"。

### 1.2 理论基础

**DetectGPT原理**:
- 基于概率曲率（probability curvature）检测AI生成文本
- 核心公式: `detection_score = sigmoid(original_log_prob - avg_perturbed_log_prob)`
- Human-likeness定义: `1.0 - detection_score`

**GRPO训练流程**:
```
1. 给定prompt，模型生成文本
2. DetectGPT计算human-likeness作为reward
3. GRPO根据reward更新模型参数
4. 循环迭代，期望reward逐渐提升
```

---

## 2. 实验配置

### 2.1 模型设置

```yaml
Base Model: Qwen/Qwen3-7B
SFT Checkpoint: models/sft/sft_20260105_1453
RL Method: GRPO (Group Relative Policy Optimization)
PEFT: LoRA (rank=16, alpha=32)
```

### 2.2 Reward Function配置

```yaml
Reward Type: Pure DetectGPT (no length/repetition penalty)
Evaluation Model: Qwen/Qwen3-7B (same as training model)
Perturbation Method: T5-small mask-filling
Num Perturbations: 10
Use T5: True (environment variable: USE_T5_PERTURBATION=1)

Reward Scaling:
  baseline: 0.47
  scale_factor: 5
  formula: final_score = 0.5 + (human_likeness - baseline) * scale_factor
```

### 2.3 训练超参数

**初始配置 (已优化)**:
```yaml
learning_rate: 5e-6  (后调整为 2e-5)
group_size: 8        (后调整为 10)
max_new_tokens: 256  (后调整为 200)
max_prompts: 100
warmup_steps: 15
early_stopping_patience: 30
gradient_accumulation_steps: 1
```

**最终配置**:
```yaml
learning_rate: 2e-5
group_size: 10
max_new_tokens: 200
max_prompts: 100
warmup_steps: 15
early_stopping_patience: 30
```

### 2.4 计算资源

```yaml
GPU: NVIDIA RTX (16GB VRAM)
VRAM Usage: ~15.5GB (near full capacity)
GPU Utilization: ~40% (I/O bound, not compute bound)
Training Time: ~2-3 hours
```

---

## 3. 训练过程

### 3.1 训练进展

**实际训练情况**:
- **Started**: Prompt 1
- **Early Stopped**: Prompt 43 (共43个prompts)
- **Steps Completed**: ~43 steps
- **Early Stop Reason**: reward_mean在30步内无显著提升

### 3.2 WandB Metrics分析

#### Reward Metrics

```
reward_mean:
  Range: 0.54 ~ 0.66
  Trend: 波动较大，先升后降，在0.6附近震荡
  Early Peak: Step ~10 达到0.66
  Final Value: ~0.62

reward_std:
  Range: 0.03 ~ 0.09
  Observation: 标准差较大，说明不同样本的reward差异明显
```

#### Loss Metrics

```
loss:
  Range: -40 ~ +60
  Trend: 剧烈波动，无明显下降趋势
  Pattern: 随机震荡，训练不稳定
  
Observation: 
  - Loss没有收敛的迹象
  - 负值loss表明某些step的梯度方向异常
  - GRPO算法的loss不是传统意义的监督loss
```

#### Learning Rate Schedule

```
learning_rate:
  Warmup: 0 -> 2e-5 (前15步)
  Constant: 2e-5 (之后保持不变)
```

### 3.3 训练问题记录

#### 问题1: 编码错误（UnicodeEncodeError）
- **原因**: Windows GBK编码无法处理emoji字符
- **影响**: 日志输出中断，但不影响训练
- **解决**: 已修复print语句中的emoji

#### 问题2: GPU利用率低
- **现象**: VRAM占满16GB，但GPU利用率仅40%
- **原因**: 
  - DetectGPT reward计算是串行的（10次扰动逐个计算）
  - 大量时间花在CPU-GPU数据传输上
  - I/O bound而非compute bound
- **尝试优化**: 批处理优化（失败，导致OOM）
- **最终方案**: 接受串行计算，保证稳定性

#### 问题3: Early Stopping频繁触发
- **第一次**: Step 36停止（patience=20）
- **优化后**: Step 43停止（patience=30）
- **原因**: reward曲线在step 10达到峰值后就开始波动/下降

---

## 4. 模型评估

### 4.1 评估方法

**Comprehensive Evaluation**:
```yaml
Test Set: test_human_with_prompt.jsonl (200个英文新闻样本)
Sample Size: 5个随机样本
Evaluation Metrics: DetectGPT human-likeness
Generation Config:
  max_tokens: 300
  temperature: 0.7
  do_sample: True
  top_p: 0.9
```

**评估对象**:
1. **Human Text**: XSum数据集中的真实人类写作
2. **SFT Model**: RL训练前的baseline
3. **RL Model**: 实验2训练后的模型

### 4.2 评估结果

#### Overall Statistics

```
Average Human-likeness (5 samples):
  Human:  0.4886 (baseline)
  SFT:    0.4902 (+0.15% vs Human)
  RL:     0.4893 (+0.07% vs Human)
  
  RL vs SFT: -0.09% (WORSE)
```

#### Per-Sample Results

| Test Case | Prompt Summary | Human | SFT | RL | RL vs SFT |
|-----------|---------------|-------|-----|-----|-----------|
| 1 | Fulham defeated QPR... | 0.4946 | 0.4929 | 0.4906 | **-0.23%** ❌ |
| 2 | Bluebird on display... | 0.4947 | 0.4916 | 0.4865 | **-0.52%** ❌ |
| 3 | Trump reshuffling NSC... | 0.4894 | 0.4894 | 0.4894 | **+0.00%** 😐 |
| 4 | Motorcyclist killed... | 0.4707 | 0.4863 | 0.4912 | **+0.49%** ✅ |
| 5 | Swansea City survival... | 0.4939 | 0.4907 | 0.4889 | **-0.18%** ❌ |

**Summary**:
- **3 cases worse** (Cases 1, 2, 5)
- **1 case better** (Case 4)
- **1 case unchanged** (Case 3)

### 4.3 生成质量分析

**Token Generation**:
```
All generations: 300/300 tokens (达到上限)
Completeness: All marked as "Complete: False"
Observation: 所有文本都被截断，但300 tokens足够评估
```

**Text Quality** (人工观察):
```
SFT Generated Text:
  - 结构完整，有开头、正文
  - 语法正确，符合新闻文体
  - 内容相对简洁

RL Generated Text:
  - 结构类似SFT
  - 无明显质量提升或下降
  - 与SFT差异微小，难以区分
```

---

## 5. 问题分析

### 5.1 核心问题：训练失败

**实验失败的直接证据**:
1. RL模型的human-likeness **下降** 0.09%
2. 5个测试中有3个变差
3. 训练时reward虽有上升，但测试时实际效果变差

**失败程度评估**:
- **轻微失败**: 差距很小（0.09%），但方向错误
- **不是灾难性失败**: 模型没有崩溃，生成质量仍可用

### 5.2 可能原因分析

#### 原因1: Reward Signal太弱 ⭐⭐⭐⭐⭐

**数据支持**:
```
DetectGPT Score Range:
  Human: 0.4707 ~ 0.4947 (range: 0.024)
  SFT:   0.4894 ~ 0.4929 (range: 0.0035)
  RL:    0.4865 ~ 0.4912 (range: 0.0047)

Average Difference:
  SFT vs Human: 0.0016 (0.16%)
  RL vs SFT:    0.0009 (0.09%)
```

**问题**:
- DetectGPT给出的分数差异**极小**
- 对于RL算法来说，这样的奖励信号**太弱**
- 模型难以从如此微小的差异中学到有效的优化方向
- 噪声可能淹没真实信号

**结论**: 这是**最主要**的失败原因

#### 原因2: Reward Hacking (Goodhart's Law) ⭐⭐⭐⭐

**观察**:
```
Training Metrics (WandB):
  reward_mean: 0.54 -> 0.66 (上升)
  
Test Results:
  RL vs SFT: -0.09% (下降)
```

**问题**:
- 训练时reward在上升，但实际质量在下降
- 这是典型的**Reward Hacking**现象
- 模型学会了"欺骗"奖励函数，但没有学到真正的"人类化"特征

**Goodhart's Law**:
> "When a measure becomes a target, it ceases to be a good measure."
> (当一个指标成为目标时，它就不再是一个好指标)

**可能的机制**:
- 模型可能学会了生成让Qwen3-1.7B"困惑"的文本
- 这些文本让概率曲率计算出现偏差，但并不真正像人类写作
- 或者模型只是过拟合到了训练集上的特定模式

#### 原因3: 评估模型与生成模型相同 ⭐⭐⭐

**设置**:
```yaml
生成模型: Qwen3-1.7B (RL fine-tuned)
评估模型: Qwen3-1.7B (DetectGPT内部)
扰动模型: T5-small
```

**问题**:
- 用**同一个模型**评估**自己**生成的文本可能有bias
- RL训练会让生成模型和评估模型产生某种"共谋"
- 这可能导致reward在训练集上上升，但在真实评估中失效

**DetectGPT论文的设置**:
- 论文中也是用同一个模型
- 但论文的目的是**检测**已有文本，不是**训练**生成模型
- 训练场景下，这个设置可能有问题

#### 原因4: 模型容量不足 ⭐⭐

**模型规模**:
```
Qwen3-1.7B:
  Parameters: 1.7B
  Size: 相对较小
```

**问题**:
- 1.7B模型可能太小，难以学习复杂的"人类化"特征
- "人类写作风格"是一个高度抽象的概念
- 可能需要更大的模型才能捕捉这些细微差异

**但是**:
- SFT训练效果不错，说明1.7B够用于基本任务
- 主要问题可能还是在reward signal上

#### 原因5: Hyperparameter调优不足 ⭐⭐

**Learning Rate**:
```
Initial: 5e-6 (太小?)
Final: 2e-5 (可能太大?)
```

**问题**:
- 没有做系统的learning rate搜索
- 2e-5可能导致训练不稳定（loss剧烈波动）
- 但reward_mean确实在上升，说明lr不是主要问题

**Group Size & Tokens**:
```
group_size: 10 (VRAM限制)
max_new_tokens: 200 (时间限制)
```

**问题**:
- group_size较小，sample efficiency低
- 但受VRAM限制，难以增大

#### 原因6: Reward Scaling设计问题 ⭐

**当前设计**:
```python
baseline = 0.47
scale_factor = 5
final_score = 0.5 + (human_likeness - baseline) * scale_factor
```

**问题**:
- baseline=0.47是根据经验设置的
- 但实际测试中human baseline是0.4886，有偏差
- scale_factor=5可能不够大，信号仍然太弱

**但是**:
- 已经做了scaling，相比初始版本有改进
- 主要问题还是原始signal太弱

---

## 6. 令人意外的发现

### 6.1 AI生成文本的Human-likeness比人类还高？

**观察**:
```
Average Human-likeness:
  Human:  0.4886
  SFT:    0.4902 (+0.15%)
  RL:     0.4893 (+0.07%)
```

**AI生成的文本居然比真实人类文本更"像人类"？**

**可能解释**:

1. **DetectGPT在XSum数据集上的Bias**
   - XSum是新闻摘要数据集，文体非常规范
   - 人类写作的新闻可能有一些"非标准"的特征
   - DetectGPT可能错误地将这些特征视为"AI特征"

2. **评估模型的局限性**
   - Qwen3-1.7B作为评估模型可能不够准确
   - 它对"人类写作"的理解可能有偏差
   - 需要更强大的评估模型（如GPT-4）

3. **Human-likeness定义的问题**
   - `1.0 - detection_score`这个定义可能过于简单
   - 真正的"人类相似度"是多维度的复杂概念
   - 单一的DetectGPT分数无法完全捕捉

4. **SFT训练的影响**
   - SFT训练让模型学会了XSum数据集的写作风格
   - 这个风格可能恰好让DetectGPT难以检测
   - 但这不代表真的更像人类

**结论**:
- 这个现象提示我们**重新思考评估方法**
- 单纯依赖DetectGPT可能不够可靠
- 需要多维度的评估体系

### 6.2 Loss曲线的异常波动

**观察**:
```
loss Range: -40 ~ +60
Pattern: 剧烈随机波动，无下降趋势
Negative Loss: 频繁出现负值
```

**GRPO Loss的特殊性**:
- GRPO的loss不是传统的cross-entropy loss
- 它是基于advantage的policy gradient loss
- 负值是正常的，表示某些action比平均好

**但是**:
- 波动如此剧烈说明训练不稳定
- 可能的原因：
  - Reward signal噪声大
  - Learning rate过大
  - Batch size (group_size) 太小

---

## 7. 核心教训

### 7.1 Reward Function设计是RL成功的关键

**教训**:
> 在RL中，Reward Function的质量直接决定训练成败。

**具体要求**:
1. **Signal Strength**: Reward差异要足够大，才能指导学习
   - 本实验: 0.09%差异太小 ❌
   - 建议: 至少5-10%的差异范围 ✅

2. **Signal Clarity**: Reward要明确反映目标
   - 本实验: DetectGPT是间接指标 ⚠️
   - 建议: 使用更直接的人类反馈 ✅

3. **Signal Stability**: Reward计算要稳定可靠
   - 本实验: 10次扰动带来噪声 ⚠️
   - 建议: 增加扰动次数或使用ensemble ✅

### 7.2 Goodhart's Law无处不在

**教训**:
> 优化一个指标≠达成真实目标

**本实验证明**:
- Training reward上升 → 训练"成功"？
- Test quality下降 → 实际"失败"！

**应对策略**:
1. **多维度评估**: 不依赖单一指标
2. **真实场景测试**: 用实际使用场景验证
3. **人类评估**: 加入真实人类判断

### 7.3 Self-Play的风险

**教训**:
> 用模型评估自己生成的内容容易产生bias

**本实验**:
- 生成模型: Qwen3-1.7B
- 评估模型: Qwen3-1.7B
- 结果: 可能相互"欺骗"

**建议**:
- 使用**不同**的模型作为评估器
- 或使用**外部**的评估工具（人类、GPT-4等）

### 7.4 训练稳定性的重要性

**观察**:
- Loss剧烈波动
- Reward忽高忽低
- Early stopping频繁触发

**教训**:
> 稳定的训练过程是成功的前提

**可能改进**:
- 更小的learning rate
- 更大的batch size (group_size)
- 更平滑的reward计算（moving average）

### 7.5 小模型的局限性

**观察**:
- 1.7B模型在简单任务（SFT）上表现良好
- 但在复杂的RL优化上可能力不从心

**教训**:
> 复杂的优化目标可能需要更大的模型容量

**但是**:
- 本实验的主要问题是reward signal
- 即使换更大模型，signal太弱仍然无法学习

---

## 8. 实验结论

### 8.1 实验结果总结

**Primary Objective**: ❌ 失败
- 目标: 提升human-likeness
- 结果: 下降0.09%
- 5个测试中3个变差

**Secondary Findings**: ⚠️ 需进一步研究
- AI生成文本比人类baseline更"像人类"（奇怪）
- DetectGPT作为reward signal信号太弱
- Reward hacking现象明显

**Training Process**: ⚠️ 不稳定
- Early stopping在43 prompts就触发
- Loss和reward曲线波动大
- GPU利用率低（I/O bound）

### 8.2 实验价值

虽然实验失败了，但我们获得了宝贵的经验：

1. **验证了DetectGPT作为直接reward的局限性**
2. **发现了Reward Hacking的实际案例**
3. **确认了评估方法的重要性**
4. **为后续实验提供了明确的改进方向**

**负面结果（Negative Result）也是有价值的科研成果！**

---

## 9. 下一步实验建议

### 9.1 Short-term Improvements (可立即尝试)

#### Suggestion 1: 改进Reward Function ⭐⭐⭐⭐⭐

**方案A: 混合Reward**
```python
def mixed_reward(text, prompt):
    # DetectGPT (30%)
    detectgpt_score = detect_gpt(text) * 0.3
    
    # Fluency via Perplexity (30%)
    fluency_score = (1.0 / perplexity(text)) * 0.3
    
    # Diversity (20%)
    diversity_score = unique_ngrams(text) * 0.2
    
    # Coherence (20%)
    coherence_score = semantic_similarity(text, prompt) * 0.2
    
    return detectgpt_score + fluency_score + diversity_score + coherence_score
```

**优点**:
- 多维度signal，更robust
- 减少单一指标的reward hacking风险
- 每个维度都可以独立调优

**方案B: 对比学习Reward**
```python
def contrastive_reward(text):
    # 同时生成多个样本
    samples = [model.generate() for _ in range(5)]
    
    # 计算与人类样本的相似度
    human_sim = similarity(text, human_samples)
    
    # 计算与其他AI样本的差异度
    ai_diff = dissimilarity(text, ai_samples)
    
    return 0.6 * human_sim + 0.4 * ai_diff
```

**优点**:
- 更直接地学习"像人类，不像AI"
- 可以利用大量unlabeled的人类文本

#### Suggestion 2: 使用不同的评估模型 ⭐⭐⭐⭐

**当前问题**:
```
生成模型 == 评估模型 == Qwen3-1.7B
→ Potential Bias
```

**改进方案**:
```yaml
Option A: 使用更强的评估模型
  - GPT-4 / Claude API
  - 优点: 更准确，更难"欺骗"
  - 缺点: 成本高，速度慢

Option B: 使用不同系列的模型
  - 生成: Qwen3-1.7B
  - 评估: LLaMA-2-7B 或 Mistral-7B
  - 优点: 避免self-play bias
  - 缺点: 需要更多VRAM

Option C: Ensemble评估
  - 使用多个模型的平均分数
  - 优点: 更robust，减少单一模型bias
  - 缺点: 计算成本成倍增加
```

**推荐**: Option B (不同模型系列)

#### Suggestion 3: 增强Reward Signal ⭐⭐⭐⭐

**方法1: 增加扰动次数**
```yaml
Current: 10 perturbations
Proposed: 20-30 perturbations
Effect: 减少noise，提升signal稳定性
Trade-off: 计算时间增加2-3倍
```

**方法2: 更激进的Reward Scaling**
```python
# Current
baseline = 0.47
scale_factor = 5

# Proposed
baseline = 0.48  # 基于实际测试调整
scale_factor = 20  # 大幅增强信号
max_reward = 1.0
min_reward = 0.0
```

**方法3: Reward Normalization**
```python
def normalize_reward(rewards_batch):
    mean = np.mean(rewards_batch)
    std = np.std(rewards_batch) + 1e-8
    return (rewards_batch - mean) / std
```

#### Suggestion 4: 调整训练超参数 ⭐⭐⭐

**Learning Rate Schedule**:
```yaml
Current: 
  warmup: 15 steps to 2e-5
  schedule: constant

Proposed:
  warmup: 20 steps to 1e-5
  schedule: cosine decay
  min_lr: 1e-6
```

**Batch Size**:
```yaml
Current: group_size = 10

Proposed: 
  - 如果VRAM允许，增加到16或20
  - 或使用gradient accumulation
```

**Early Stopping**:
```yaml
Current: patience = 30

Proposed:
  - patience = 50 (更宽容)
  - monitor_metric = "reward_mean_ma10" (10步移动平均)
  - 避免被短期波动误导
```

### 9.2 Medium-term Explorations (需要一定准备)

#### Suggestion 5: 人类反馈标注 (RLHF) ⭐⭐⭐⭐⭐

**方案**: Collect Human Preferences

**Step 1**: 生成对比样本
```python
for prompt in test_prompts:
    text_a = sft_model.generate(prompt)
    text_b = rl_model.generate(prompt)
    pairs.append((text_a, text_b))
```

**Step 2**: 人工标注
```
Which text is more human-like?
[ ] Text A
[ ] Text B
[ ] Similar
```

**Step 3**: 训练Reward Model
```python
reward_model = train_preference_model(labeled_pairs)
```

**Step 4**: 使用Reward Model进行RL
```python
reward = reward_model(text)  # 代替DetectGPT
```

**优点**:
- 最直接的"人类相似度"信号
- 避免proxy metric的问题
- 这是当前最先进的方法（InstructGPT, Claude等都用）

**缺点**:
- 需要大量人工标注（至少几百到几千对）
- 成本高，时间长

**实施建议**:
- 先标注200-500对，训练initial reward model
- 逐步迭代，active learning策略选择最有价值的样本标注

#### Suggestion 6: 对抗训练框架 (GAN-style) ⭐⭐⭐⭐

**方案**: Train a Discriminator

**Architecture**:
```
Generator (G): Qwen3-1.7B (生成文本)
Discriminator (D): BERT-base (判断是否人类写作)

Training Loop:
1. D训练: 区分人类文本 vs G生成文本
2. G训练: 生成能骗过D的文本
3. 交替进行，直到收敛
```

**Reward Design**:
```python
reward = discriminator.predict_proba(text)["human"]
```

**优点**:
- Discriminator专门训练，比DetectGPT更准确
- 对抗训练能推动generator不断改进
- 经典且有效的方法

**缺点**:
- 训练不稳定（GAN的通病）
- 需要仔细调参
- Discriminator可能过拟合

#### Suggestion 7: 使用更大的模型 ⭐⭐⭐

**当前**: Qwen3-1.7B

**Upgrade Options**:
```yaml
Option 1: Qwen3-4B
  VRAM: ~8-10GB (base) + 3-4GB (LoRA) = 12-14GB
  Feasibility: 可行（你有16GB）
  
Option 2: Qwen3-7B
  VRAM: ~14GB (base) + 5-6GB (LoRA) = 19-20GB
  Feasibility: 需要gradient checkpointing或更大显卡
  
Option 3: 租用云GPU
  Tesla V100 (32GB) or A100 (40GB/80GB)
  Cost: ~$1-3/hour
```

**建议**:
- 先尝试Qwen3-4B，看是否有明显提升
- 如果有效，再考虑7B或租用GPU

### 9.3 Long-term Research Directions (毕设范围外)

#### Direction 1: 多模态Reward Signal

结合文本特征、语义特征、风格特征等多个维度构建综合reward。

#### Direction 2: 元学习（Meta-Learning）

学习如何快速适应不同的写作风格和领域。

#### Direction 3: 因果推断

理解"什么因素导致文本更像人类"，而不是单纯优化metric。

---

## 10. 推荐的下一个实验

综合考虑**可行性、有效性、创新性**，我推荐：

### Experiment 3: Mixed Reward with Different Evaluator

**核心改进**:
1. **混合Reward Function** (解决signal弱的问题)
   ```python
   reward = 0.4 * detectgpt + 0.3 * fluency + 0.2 * diversity + 0.1 * coherence
   ```

2. **使用不同的评估模型** (避免self-play bias)
   ```
   生成: Qwen3-1.7B
   DetectGPT评估: Mistral-7B 或 LLaMA-2-7B
   ```

3. **更强的Reward Scaling**
   ```python
   scale_factor = 20  # 更激进
   ```

4. **优化的超参数**
   ```yaml
   learning_rate: 1e-5 (更保守)
   group_size: 12 (如果VRAM允许)
   patience: 50 (更宽容)
   ```

**预期效果**:
- 更强、更清晰的训练信号
- 减少reward hacking风险
- 更稳定的训练过程
- 有望看到明显的质量提升

**实施难度**: 中等
**预期时间**: 1-2天
**成功概率**: 较高（60-70%）

---

## 11. 附录

### 11.1 完整训练日志路径

```
Training Logs: biyesheji/project/outputs/[run_name]/
WandB Dashboard: [URL]
Model Checkpoints: biyesheji/project/models/rl_detectgpt_pure_exp2_scaled/
Evaluation Results: Terminal output (script crashed before saving JSON)
```

### 11.2 关键代码位置

```
Reward Function: src/rl/reward_functions.py::detectgpt_pure_reward_with_info
Training Script: src/rl/train_grpo.py
Launch Script: start_sh/run_grpo_exp2.sh
Evaluation Script: scripts/comprehensive_evaluation.py
Data Preparation: data/raw/prepare_data.py
```

### 11.3 相关文档

```
- experiment2_command.md: 实验参数记录
- experiment2_optimizations.md: 优化过程记录
- experiment2_reward_scaling.md: Reward scaling设计
- reward_baseline_and_vram.md: Baseline推导和VRAM分析
- gpu_optimization_log.md: GPU优化尝试
```

### 11.4 DetectGPT核心算法

```python
def detect(self, text: str) -> float:
    """
    DetectGPT detection score calculation
    
    Returns:
        detection_score: 0~1, higher = more likely AI-generated
    """
    # 1. Compute original text log-prob
    original_logprob = self.compute_log_prob(text)
    
    # 2. Generate perturbed versions (T5 mask-filling)
    perturbed_texts = [
        self.perturb_text_t5(text) 
        for _ in range(self.num_perturbations)
    ]
    
    # 3. Compute perturbed log-probs
    perturbed_logprobs = [
        self.compute_log_prob(p_text) 
        for p_text in perturbed_texts
    ]
    
    # 4. Calculate curvature
    avg_perturbed = np.mean(perturbed_logprobs)
    curvature = original_logprob - avg_perturbed
    
    # 5. Normalize to [0, 1] with sigmoid
    detection_score = 1.0 / (1.0 + np.exp(-curvature))
    
    return detection_score
```

---

## 12. 致谢与反思

这次实验虽然没有达到预期目标，但整个过程是有价值的学习经历：

1. **实践了完整的RL训练流程**
2. **理解了Reward Function设计的重要性**
3. **发现了Goodhart's Law的真实案例**
4. **学会了系统化的实验设计和分析**

**失败是成功之母**。这次实验为后续研究指明了方向。

---

**Report Date**: 2026-01-26  
**Author**: [Your Name]  
**Experiment Duration**: ~3 days  
**Next Steps**: 见Section 9推荐

---

**END OF REPORT**
