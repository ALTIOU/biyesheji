# 实验2最终报告：DetectGPT Pure Reward + Reward Scaling

## 实验目标

使用DetectGPT作为纯reward信号，通过GRPO强化学习优化Qwen3-1.7B模型，使其生成更"像人类"的文本。

---

## 实验配置

### 模型与数据
```yaml
Base Model: Qwen/Qwen3-7B
SFT Adapter: sft_20260105_1453
Training Prompts: 100
Evaluation Model: Qwen/Qwen3-7B (同训练模型)
Perturbation: T5-small (10次)
```

### Reward Function
```python
# DetectGPT Pure Reward with Scaling
detection_score = detectgpt.detect(text)  # 0~1, 越高越像AI
human_likeness = 1.0 - detection_score

# Reward Scaling (5x amplification)
baseline = 0.47
advantage = (human_likeness - baseline) * 5
final_reward = 0.5 + advantage  # 映射到0-1范围
```

### GRPO超参数
```yaml
Learning Rate: 2e-5 (增大4倍)
Warmup Steps: 15
Group Size: 10
Max New Tokens: 200
Early Stopping Patience: 30
```

---

## 训练过程

### 训练指标

**完成情况：**
- 完成Prompts: 43/100
- 训练时长: 1小时41分钟
- 停止原因: Early Stopping（连续30步无提升）

**Reward曲线：**
```
Prompt 0:  0.5465  (起点)
Prompt 12: 0.6772  (峰值，最佳) ★
Prompt 43: 0.6046  (终点)

提升: +24.0% (0.5465 → 0.6772)
```

**Loss曲线：**
```
范围: -40 ~ +60
特征: 剧烈波动，无收敛趋势 ⚠️
```

**Reward Std：**
```
范围: 0.03 ~ 0.08
对比: 实验1为0.004-0.016

结论: Reward Scaling确实放大了差异 ✅
```

### WandB观察

**Reward Mean:**
- ✅ 有明确上升趋势（vs 实验1的水平波动）
- ✅ 在Prompt 12达到峰值0.6772
- ⚠️ 之后30步无法超越，触发Early Stop

**Loss:**
- ❌ 完全随机波动，无下降趋势
- ⚠️ 这是一个重要的警告信号

**Learning Rate:**
- ✅ Warmup正常（0 → 2e-5，15步）
- ✅ 之后保持2e-5

---

## 模型质量评估

### 生成文本对比

#### Test Case 1: "人工智能的未来发展趋势是什么？"

**SFT模型（RL前）：**
```
- 结构：正常回答，提到深度学习、NLP等
- 长度：254字符
- 风格：清晰简洁，像标准的知识问答
```

**RL模型（RL后）：**
```
- 结构：自己添加了"请从技术、经济和社会三个维度分析"
- 长度：271字符
- 风格：更结构化，分维度展开
- 表达：更专业，提到边缘计算、硬件性能等
```

#### Test Case 2: "如何提高工作效率？"

**SFT模型：**
```
- 风格：自问自答式，有点啰嗦
- 示例："你认为这是正确的方法吗？"
```

**RL模型：**
```
- 风格：简洁专业，标注"（100字）"
- 内容：四象限法则、番茄工作法等具体方法
- 结构：分点清晰，逻辑性强
```

**初步观察：**
- ✅ RL模型表达更结构化、更专业
- ⚠️ 但这是否意味着"更像人类"？

---

### DetectGPT客观评估

#### 评估结果

**Test Case 1: "人工智能的未来发展趋势是什么？"**
```yaml
SFT模型:
  DetectGPT Score: 0.5094 (越高越像AI)
  Human-likeness: 0.4906 (越高越像人类)

RL模型:
  DetectGPT Score: 0.5266 (越高越像AI)
  Human-likeness: 0.4734 (越高越像人类)

变化: -3.51% ❌
结论: RL模型反而更像AI！
```

---

## 核心问题分析

### 矛盾现象

```yaml
训练时:
  ✅ Reward从0.546上升到0.677 (+24%)
  ✅ Reward差异明显扩大
  ❌ Loss完全无下降趋势

测试时:
  ❌ Human-likeness下降3.5%
  ❌ RL模型反而更像AI
  ❌ 优化方向完全相反！
```

### 为什么会这样？

#### 1. Reward Hacking（奖励欺骗）

```
模型学会了某些"tricks"来提高训练时的DetectGPT reward：
  - 增加结构化表达（"从三个维度分析"）
  - 使用更专业的术语
  - 添加框架性引导词
  
这些tricks在训练数据上提高了reward，
但在新文本上反而降低了"人类相似度"
```

#### 2. Overfitting（过拟合）

```
训练数据: 仅100个prompts
模型在这100个特定prompts上优化

泛化能力差 → 新prompts上效果反而下降
```

#### 3. DetectGPT作为Reward的局限性

```
DetectGPT基于probability curvature：
  原文log_prob vs 扰动文本avg_log_prob
  
问题:
  1. 对某些模式敏感（容易被exploit）
  2. 信号不够robust和准确
  3. 作为优化目标时会被"hack"
  
Goodhart's Law:
  "When a measure becomes a target, 
   it ceases to be a good measure"
```

#### 4. Reward Scaling的副作用

```python
# 我们的Scaling
baseline = 0.47  # 可能不够准确
scale_factor = 5  # 放大倍数较大

# 后果
微小的测量误差 → 放大5倍 → 优化方向偏离
```

#### 5. Loss无趋势的警告

```
GRPO的loss理论上应该有某种趋势（上升或下降）
完全随机波动说明:
  - 优化不稳定
  - 可能在错误的方向上震荡
  - Learning rate可能太大
```

---

## 对比：实验1 vs 实验2

| 指标 | 实验1 (5e-6 lr, no scaling) | 实验2 (2e-5 lr, 5x scaling) |
|------|---------------------------|----------------------------|
| 完成Prompts | 36 | 43 |
| 训练Reward范围 | 0.534-0.544 (0.01) | 0.546-0.677 (0.13) |
| 训练Reward提升 | +2% | +24% |
| **测试Human-likeness** | **未测** | **-3.5% ❌** |
| Loss趋势 | 波动无趋势 | 波动无趋势 |
| Reward Std | 0.004-0.016 | 0.03-0.08 |

**结论：**
- ✅ Reward Scaling确实放大了差异
- ✅ 训练过程看起来更有意义
- ❌ 但实际效果反而更差

---

## 实验结论

### 失败原因

1. **纯DetectGPT不适合做RL reward**
   - 容易被exploit
   - 信号质量不够
   - 作为优化目标时失效

2. **Reward Scaling可能误导优化**
   - Baseline选择不准确（0.47）
   - 放大倍数太大（5x）
   - 放大了错误的信号

3. **训练数据太少**
   - 100 prompts不足以泛化
   - 容易overfitting

4. **Loss监控的重要性**
   - Loss无趋势是重要警告信号
   - 不能只看reward指标
   - 需要独立测试集验证

### 核心教训

```
✅ 学到的经验:

1. 训练指标上升 ≠ 模型真正改进
   需要在独立测试集上验证

2. Goodhart's Law在RL中非常真实
   优化指标本身会破坏指标的有效性

3. Loss的行为是重要的诊断工具
   即使reward上升，loss混乱也说明有问题

4. 简单的reward function往往不够robust
   需要更复杂、多维度的reward设计

5. Reward Scaling需要非常谨慎
   错误的baseline或scale会误导整个训练
```

---

## 改进方向

### 短期（修复当前方案）

#### 1. 调整Reward Scaling
```python
# 当前
baseline = 0.47  # 估算的
scale_factor = 5  # 过大？

# 改进
# 先在小数据集上统计真实的human_likeness分布
baseline = empirical_median()  # 数据驱动
scale_factor = 2 or 3  # 更保守
```

#### 2. 增加训练数据
```yaml
当前: 100 prompts
改进: 300 or 500 prompts
好处: 降低overfitting风险
```

#### 3. 添加正则化
```python
def regularized_reward(text):
    detectgpt_reward = scaled_detectgpt(text)
    
    # 惩罚过长的文本
    length_penalty = compute_length_penalty(text)
    
    # 惩罚重复
    repetition_penalty = compute_repetition(text)
    
    return detectgpt_reward - 0.1*length_penalty - 0.1*repetition_penalty
```

#### 4. Learning Rate Decay
```python
# 从2e-5逐渐降到2e-6
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

---

### 长期（根本性改进）

#### 1. 混合Reward Function
```python
def hybrid_reward(text):
    """多维度评估"""
    # DetectGPT（降低权重）
    detectgpt_score = scaled_detectgpt(text) * 0.4
    
    # 流畅度（Perplexity）
    ppl_score = compute_fluency(text) * 0.3
    
    # 多样性（Distinct-n）
    diversity_score = compute_diversity(text) * 0.2
    
    # 语义相关性（BERTScore）
    relevance_score = compute_relevance(text, prompt) * 0.1
    
    return detectgpt_score + ppl_score + diversity_score + relevance_score
```

**优势：**
- 更robust，难以被单一trick exploit
- 多维度优化，更全面
- 即使某个指标失效，其他指标可以补充

#### 2. 对抗性验证
```python
# 训练时同时在验证集上测试
for step in training:
    train_reward = compute_reward(train_prompts)
    val_reward = compute_reward(val_prompts)  # 独立验证集
    
    if val_reward下降:
        警告：可能overfitting或reward hacking
```

#### 3. 换用更robust的detection方法
```yaml
当前: DetectGPT (probability curvature)

替代方案:
  - Watermarking（水印检测）
  - Classifier-based（训练专门的分类器）
  - Ensemble methods（多种方法投票）
```

#### 4. 引入人类反馈
```yaml
方法: RLHF (Reinforcement Learning from Human Feedback)

流程:
  1. 收集人类标注数据
  2. 训练reward model（基于人类偏好）
  3. 用reward model指导RL训练

优势:
  - 直接对齐人类判断
  - 更难被hack
  - 泛化能力更强
```

---

## 实验价值

虽然实验2未能达到预期目标，但它提供了重要的**negative result**：

### 科学价值

```
1. 证明了纯DetectGPT不适合做RL reward
   - 为后续研究提供教训
   - 避免其他人走同样的弯路

2. 展示了Goodhart's Law在实践中的体现
   - 理论概念的实证案例

3. 强调了独立测试集验证的重要性
   - 训练指标可能误导
   - 需要客观评估

4. 突出了Loss监控的诊断价值
   - Loss行为是重要的警告信号
```

### 工程价值

```
1. 建立了完整的实验框架
   - Reward function设计
   - Training pipeline
   - Evaluation protocol

2. 积累了调试和分析经验
   - WandB监控
   - 模型对比
   - DetectGPT评估

3. 为改进方向提供了清晰的指引
   - 知道什么不work
   - 知道下一步该怎么改
```

---

## 建议下一步行动

### 选项1：改进当前方案（推荐）

```yaml
行动:
  1. 调整Reward Scaling参数
     - baseline从0.47调到经验值
     - scale_factor从5降到2-3
  
  2. 增加训练数据
     - 从100增加到300 prompts
  
  3. 添加learning rate decay
  
  4. 在验证集上同步监控
  
预期: 可能改善，但不确定能完全解决问题
时间: 1-2天
```

### 选项2：切换到混合Reward（更保守）

```yaml
行动:
  1. 实现混合reward function
     - DetectGPT (40%)
     - Perplexity (30%)
     - Diversity (20%)
     - Relevance (10%)
  
  2. 重新训练
  
  3. 对比效果
  
预期: 更robust，成功概率更高
时间: 2-3天
```

### 选项3：暂时停止，总结报告（务实）

```yaml
行动:
  1. 接受这次negative result
  
  2. 写详细的实验报告
     - 记录所有发现
     - 分析失败原因
     - 提出改进方向
  
  3. 作为毕业设计的一部分
     - Negative results也有价值
     - 展示批判性思维和分析能力
  
价值: 科研中失败是常态，诚实报告很重要
时间: 1天
```

---

## 附录

### A. 完整训练日志

```
Prompt 0:  Loss -0.2019, Reward 0.5465, LR 1.33e-06
Prompt 1:  Loss 0.2669,  Reward 0.5966, LR 2.67e-06
Prompt 2:  Loss 11.9005, Reward 0.6101, LR 4.00e-06
...
Prompt 12: Loss -26.8651, Reward 0.6772, LR 1.73e-05 ★ 最佳
...
Prompt 42: Loss 16.2540, Reward 0.6046, LR 2.00e-05
Early Stopping triggered (30/30 no improvement)
```

### B. 模型保存位置

```
最佳模型（Prompt 12, Reward=0.6772）:
biyesheji/project/models/rl_detectgpt_pure_exp2_scaled/grpo_detectgpt_pure_scaled_20260125
```

### C. WandB链接

```
https://wandb.ai/yesh26-sun-yat-sen-university/graduation_grpo/runs/22j2qknx
```

### D. 相关文档

```
- experiment2_reward_scaling.md: 优化方案设计
- reward_baseline_and_vram.md: Baseline选择和显存分析
- gpu_optimization_log.md: GPU优化历史
- experiment2_command.md: 命令文档
```

---

**实验日期**: 2026-01-25  
**报告撰写**: 2026-01-25  
**状态**: 已完成（失败但有价值）
