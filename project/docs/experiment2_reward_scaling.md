# 实验2 - Reward Scaling优化版

## 问题诊断

### 原始训练问题（36步后Early Stopping）

**WandB观察结果：**
```yaml
reward_mean: 0.534 ~ 0.544 (仅0.01变化)
reward_std: 0.004 ~ 0.016 (波动很小)
loss: -20 ~ +40 (剧烈波动，无收敛趋势)
learning_rate: 0 → 5e-6 (warmup正常)
```

**根本原因分析：**

1. **Reward信号太弱** ⚠️
   ```python
   DetectGPT detection_score: 0.5 ~ 0.6
   → human_likeness: 0.4 ~ 0.5
   → final_score: 0.44 ~ 0.55 (仅0.11范围)
   
   实际观察到的reward_mean变化：
   最小: 0.534
   最大: 0.544
   差异: 0.01 (仅2%变化！)
   ```

   **问题：**
   - 不同生成质量的文本，reward差异太小
   - RL算法无法有效区分"好"和"坏"的生成
   - 导致policy gradient接近随机游走

2. **Learning Rate过小**
   ```
   当前: 5e-6
   问题: 即使有明确的gradient，更新步长太小
   导致: 36步仍无明显改进
   ```

3. **Loss剧烈波动**
   ```
   GRPO的policy gradient方差大
   group_size=8可能太小
   → 估计不准确 → loss上下震荡
   ```

4. **Early Stopping过于严格**
   ```
   patience=20步
   问题: reward变化太小，很容易触发
   实际: 36步就停止，可能训练不充分
   ```

---

## 优化方案

### 1. **Reward Scaling（核心改进）**

**改进前：**
```python
detection_score = detector.detect(text)      # 0.5-0.6
human_likeness = 1.0 - detection_score       # 0.4-0.5  
final_score = human_likeness * 1.1           # 0.44-0.55
```

**改进后：**
```python
detection_score = detector.detect(text)      # 0.5-0.6
human_likeness = 1.0 - detection_score       # 0.4-0.5

# Reward Scaling: 放大微小差异
baseline = 0.47                              # 中位数
advantage = (human_likeness - baseline) * 5  # 放大5倍！
final_score = 0.5 + advantage                # 重新映射
```

**效果对比：**
```yaml
改进前（原始reward）:
  human_likeness=0.44 → final=0.484 → reward_mean≈0.534
  human_likeness=0.47 → final=0.517 → reward_mean≈0.540
  human_likeness=0.50 → final=0.550 → reward_mean≈0.544
  
  差异: 0.544 - 0.534 = 0.01 (太小！)

改进后（5x scaling）:
  human_likeness=0.44 → advantage=-0.15 → final=0.35
  human_likeness=0.47 → advantage=0.00  → final=0.50
  human_likeness=0.50 → advantage=0.15  → final=0.65
  
  差异: 0.65 - 0.35 = 0.30 (30倍放大！)
```

**为什么有效？**
- RL算法对**相对差异**敏感，不是绝对值
- 放大差异 → 更清晰的学习信号
- 模型能更明确地知道"什么生成更好"

---

### 2. **增大Learning Rate**

```yaml
改进前: 5e-6 (太保守)
改进后: 2e-5 (增大4倍)
```

**原因：**
- 原lr过小，即使有明确gradient也更新缓慢
- 配合放大的reward，需要更大的步长才能有效学习
- 从WandB看loss波动大但无趋势，说明需要更aggressive的更新

**Risk vs Benefit：**
- ⚠️ Risk: 可能不稳定，loss震荡更大
- ✅ Benefit: 更快收敛，明确的学习趋势
- 🎯 Strategy: 搭配更长的warmup（15步）和更宽容的early stopping（30步）

---

### 3. **调整GRPO超参数**

```yaml
group_size:
  改进前: 8  (小batch，高variance)
  改进后: 12 (中等batch，平衡variance和计算)

max_new_tokens:
  改进前: 256 (长生成，慢)
  改进后: 200 (合理长度，快)

warmup_steps:
  改进前: 10
  改进后: 15 (给更大lr更长的warm up时间)

early_stopping_patience:
  改进前: 20 (太严格，36步就停了)
  改进后: 30 (更宽容，给模型更多探索时间)
```

**权衡分析：**
```
group_size=12:
  ✅ 更稳定的gradient估计（减少loss波动）
  ✅ 仍然能装进16GB显存
  ⚠️ 每步稍慢（12个样本 vs 8个）

max_new_tokens=200:
  ✅ 每步更快（256→200 tokens）
  ✅ 仍然足够长（多数文本<200 tokens）
  ⚠️ 极少数需要256 tokens的文本会被截断

patience=30:
  ✅ 避免过早停止
  ✅ 给reward scaling足够时间展现效果
  ⚠️ 如果确实无效，会浪费更多时间（但30步不算多）
```

---

## 预期效果

### 训练曲线预期变化

**Reward Mean：**
```
改进前: 0.534-0.544 水平波动，无趋势
改进后: 0.30-0.70 明确上升或下降趋势
```

**Loss：**
```
改进前: -20~+40 剧烈波动
改进后: 可能初期波动仍大，但应该有总体趋势（上升或下降）
       如果loss仍然只是波动无趋势 → 说明问题不在reward scaling
```

**Reward Std：**
```
改进前: 0.004-0.016 (极小)
改进后: 0.05-0.20 (明显增大，说明同一batch内样本差异被放大了)
```

---

## 成功标准

### 本次实验目标

**主要目标：**
1. ✅ Reward mean有明确的上升或下降趋势（不再是水平波动）
2. ✅ Loss有收敛或发散的趋势（不再是无方向波动）
3. ✅ 至少训练50+ prompts（不要36步就early stop）

**次要目标：**
4. Reward mean上升到0.6+（说明模型在学习生成更"人类化"的文本）
5. 训练能跑完100个prompts，或在50-80步之间合理early stop

### 如何判断实验成功？

**场景A：Reward上升 + Loss收敛** ✅✅✅
```
说明：模型成功学习到了"如何生成更像人类的文本"
结论：Reward scaling有效！DetectGPT可以作为RL信号
下一步：继续训练更多steps，或调优到更大数据集
```

**场景B：Reward下降 + Loss收敛** ⚠️ 但也有意义
```
说明：模型学习到了一些东西，但优化方向可能反了
可能原因：
  - Reward scaling的baseline不准（0.47可能不是真实中位数）
  - DetectGPT本身有bias
结论：需要调整baseline或重新审视reward函数
但至少说明：Reward差异放大后，模型能学习！
```

**场景C：Reward波动 + Loss仍然无趋势** ❌
```
说明：Reward scaling不是主要问题，可能：
  - DetectGPT本身信号质量差（perturbation不够好）
  - GRPO算法不适合这个任务
  - 需要更根本的改变（如换PPO、换reward函数）
结论：放弃纯DetectGPT，考虑混合reward或其他方法
```

---

## 启动命令

### PowerShell命令（Windows）

```powershell
# 设置环境变量
$env:USE_T5_PERTURBATION = "1"

# 激活conda环境（如果需要）
# conda activate biyeshejihuanjing

# 运行训练
python biyesheji/project/src/rl/train_grpo.py `
  --base_model Qwen/Qwen3-7B `
  --sft_adapter_path biyesheji/project/models/sft/sft_20260105_1453 `
  --rl_data_path biyesheji/project/data/processed/rl_prompts.jsonl `
  --output_dir biyesheji/project/models/rl_detectgpt_pure_exp2_scaled `
  --reward_name detectgpt_pure `
  --max_prompts 100 `
  --group_size 12 `
  --max_new_tokens 200 `
  --learning_rate 2e-5 `
  --warmup_steps 15 `
  --early_stopping_patience 30 `
  --report_to wandb `
  --wandb_project graduation_grpo `
  --run_name grpo_detectgpt_pure_scaled_20260125
```

### 估算时间

```yaml
配置:
  prompts: 100
  group_size: 12
  max_new_tokens: 200
  perturbations: 10
  
预估:
  每个prompt时间: 
    - 生成12个样本: ~30s (200 tokens each)
    - 计算12个DetectGPT score (10次perturbation each): ~4 min
    - GRPO更新: ~10s
    - 总计: ~5 min/prompt
  
  总时间: 100 prompts × 5 min = ~8 hours
  
  (如果early stop在60步: ~5 hours)
```

---

## 监控要点

### 训练过程中关注

1. **前15步（Warmup阶段）**
   ```
   关注: Learning rate是否平滑增长 0→2e-5
   关注: Reward是否开始出现方向性变化
   ```

2. **15-30步（初期学习）**
   ```
   关注: Reward mean是否有明确趋势（上升/下降）
   关注: Loss是否开始收敛
   关注: Reward std是否比之前大（应该从0.01→0.1左右）
   ```

3. **30-60步（稳定阶段）**
   ```
   关注: 趋势是否持续
   关注: 是否有overfitting迹象（reward上升但std下降）
   ```

4. **60+步（后期）**
   ```
   关注: 是否触发early stopping
   关注: 最佳reward值达到多少
   ```

### WandB关键指标

```yaml
必看图表:
  1. reward_mean vs step  (最重要！看趋势)
  2. loss vs step         (看是否收敛)
  3. reward_std vs step   (应该比之前大)
  4. learning_rate vs step (检查warmup)

次要图表:
  5. parts/detection_score (DetectGPT原始分数)
  6. parts/human_likeness  (1-detection_score)
  7. parts/advantage       (放大后的差异，新增的！)
```

---

## 如果还是不work？

### 备选方案B：混合Reward

如果纯DetectGPT + Reward Scaling仍然无效（reward无趋势、loss不收敛），考虑：

```python
def detectgpt_hybrid_reward(text: str) -> float:
    """
    混合多种信号，避免单一信号太弱
    """
    # 1. DetectGPT (scaled): 权重0.6
    detectgpt_score = scaled_detectgpt(text) * 0.6
    
    # 2. 流畅度 (perplexity): 权重0.2
    fluency_score = compute_fluency(text) * 0.2
    
    # 3. 长度合理性: 权重0.1
    length_score = compute_length_score(text) * 0.1
    
    # 4. 多样性 (distinct-n): 权重0.1
    diversity_score = compute_diversity(text) * 0.1
    
    return detectgpt_score + fluency_score + length_score + diversity_score
```

### 备选方案C：增加Perturbation次数

```python
# 当前: 10次perturbation
# 改为: 20次或30次
# 
# 优点: DetectGPT分数更稳定，噪音更少
# 缺点: 训练变慢2-3倍
```

### 备选方案D：检查T5 Perturbation质量

```python
# 打印几个perturbation样本，人工检查：
# - T5是否真的在做mask-filling？
# - Perturbation是否保留原意？
# - Perturbation是否足够多样？
```

---

## 实验记录模板

```yaml
实验名称: Experiment 2 - Reward Scaling
开始时间: 2026-01-25 [填写实际时间]
WandB Run: grpo_detectgpt_pure_scaled_20260125

配置:
  Reward: detectgpt_pure (with 5x scaling)
  Learning Rate: 2e-5 (warmup 15 steps)
  Group Size: 12
  Max Tokens: 200
  Perturbations: 10 (T5-small)
  Early Stop: 30 patience
  Max Prompts: 100

结果:
  停止步数: [填写]
  停止原因: [Early Stop / 完成 / 手动停止 / OOM]
  最佳Reward: [填写]
  最终Loss: [填写]
  训练时长: [填写]

WandB观察:
  Reward趋势: [上升/下降/波动无趋势]
  Loss趋势: [收敛/发散/波动无趋势]
  Reward Std: [填写范围]

结论:
  成功？ [是/否/部分成功]
  Reward Scaling有效？ [是/否]
  下一步: [填写计划]
```

---

## 参考资料

### DetectGPT原论文

- **标题**: DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature
- **方法**: 
  - Perturbation: T5 mask-filling, 100次
  - 评估: 计算原文与扰动文本的log-prob差异
- **我们的实现差异**:
  - Perturbation次数: 10次 (vs 论文100次)
  - Reward scaling: 5倍放大 (论文没有用于RL)

### GRPO相关

- **Group Relative Policy Optimization**
- **特点**: 使用group内的相对reward，而不是absolute reward
- **问题**: 对reward signal的scale敏感性较低（理论上）
- **实践**: 我们发现仍然需要足够大的reward差异才能有效学习

---

**更新日期**: 2026-01-25  
**下次更新**: 本次训练结束后，填写结果
