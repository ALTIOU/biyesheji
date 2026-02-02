# Reward Baseline 和 显存配置说明

## 1. Reward Baseline = 0.47 的来源

### 计算过程

**观察数据（实验2第一次运行）：**
```yaml
reward_mean: 0.534 ~ 0.544
reward_std: 0.004 ~ 0.016
```

**反推human_likeness：**

在旧的reward函数中：
```python
# 旧版本
human_likeness = 1.0 - detection_score
final_score = human_likeness * 1.1
```

从reward_mean反推：
```
reward_mean ≈ 0.540
→ human_likeness * 1.1 = 0.540
→ human_likeness = 0.540 / 1.1 ≈ 0.491
```

考虑全范围（0.534-0.544）：
```
下限: 0.534 / 1.1 ≈ 0.485
上限: 0.544 / 1.1 ≈ 0.495
中位数: (0.485 + 0.495) / 2 ≈ 0.490
```

**为什么用0.47而不是0.49？**
- 稍微保守一点，偏向下界
- 0.47对应detection_score = 0.53，这是一个常见的"中性"值
- 如果设为0.49，可能会使大部分样本的advantage都是负数

---

## 2. Baseline的影响分析

### 不同baseline的效果

```python
# 假设human_likeness范围是0.44-0.50

# Baseline = 0.45 (偏低)
human_likeness=0.44 → advantage=(0.44-0.45)*5=-0.05 → final=0.45
human_likeness=0.47 → advantage=(0.47-0.45)*5=+0.10 → final=0.60
human_likeness=0.50 → advantage=(0.50-0.45)*5=+0.25 → final=0.75
→ 结果：大部分样本都是正reward，模型倾向于认为"都不错"

# Baseline = 0.47 (中等，当前设置)
human_likeness=0.44 → advantage=(0.44-0.47)*5=-0.15 → final=0.35
human_likeness=0.47 → advantage=(0.47-0.47)*5=0.00  → final=0.50
human_likeness=0.50 → advantage=(0.50-0.47)*5=+0.15 → final=0.65
→ 结果：对称分布，有好有坏，差异明显

# Baseline = 0.49 (偏高)
human_likeness=0.44 → advantage=(0.44-0.49)*5=-0.25 → final=0.25
human_likeness=0.47 → advantage=(0.47-0.49)*5=-0.10 → final=0.40
human_likeness=0.50 → advantage=(0.50-0.49)*5=+0.05 → final=0.55
→ 结果：大部分样本都是负reward，模型倾向于认为"都不好"
```

### 如何判断baseline是否合理？

**训练开始后，在WandB观察：**

1. **parts/advantage** 分布（新增的metric）：
   ```
   理想情况: -0.2 到 +0.2，均匀分布
   baseline太低: 大部分advantage都是正数
   baseline太高: 大部分advantage都是负数
   ```

2. **reward_mean** 趋势：
   ```
   如果baseline准确：reward_mean应该在0.4-0.6范围波动
   如果baseline太低：reward_mean会偏高（>0.6），且很难下降
   如果baseline太高：reward_mean会偏低（<0.4），且很难上升
   ```

3. **模型学习效果**：
   ```
   baseline合理：模型能区分好坏，reward有明确趋势
   baseline不合理：reward波动，无法收敛
   ```

---

## 3. 调整Baseline（如果需要）

### 方法1：通过环境变量

```powershell
# 默认（当前）
$env:REWARD_BASELINE = "0.47"
$env:REWARD_SCALE_FACTOR = "5"

# 如果发现baseline太高，可以降低
$env:REWARD_BASELINE = "0.45"

# 如果发现baseline太低，可以增加
$env:REWARD_BASELINE = "0.49"

# 如果发现放大倍数太大（reward变化太剧烈），可以降低
$env:REWARD_SCALE_FACTOR = "3"
```

### 方法2：训练前先采样验证

**理想做法（如果有时间）：**
```python
# 生成10-20个样本，计算它们的human_likeness
# 取中位数作为baseline

samples = [...] # 从训练数据随机采样
human_likeness_values = []
for sample in samples:
    detection_score = detector.detect(sample)
    human_likeness = 1.0 - detection_score
    human_likeness_values.append(human_likeness)

baseline = np.median(human_likeness_values)
print(f"Empirical baseline: {baseline:.3f}")
```

---

## 4. 显存配置详解

### 显存占用构成（16GB GPU）

```yaml
固定占用（加载模型时）:
  1. 训练模型 Qwen3-1.7B + LoRA:
     - Base model (FP16): ~3.4 GB
     - LoRA parameters: ~0.5 GB
     - 优化器状态 (LoRA only): ~1.5 GB
     小计: ~5.4 GB

  2. DetectGPT评估模型 Qwen3-1.7B:
     - Base model (FP16): ~3.4 GB
     - 推理缓存: ~0.1 GB
     小计: ~3.5 GB

  3. T5-small扰动模型:
     - Model: ~240 MB
     - 推理缓存: ~10 MB
     小计: ~0.25 GB

  固定占用总计: ~9.15 GB

动态占用（训练时）:
  1. 生成阶段（前向传播）:
     - KV cache: group_size × max_new_tokens × layers × hidden
     - 8 × 256: ~2.5 GB
     - 10 × 200: ~2.4 GB
     - 12 × 200: ~2.9 GB

  2. GRPO计算（reward, advantage, loss）:
     - 中间张量: group_size × max_new_tokens × features
     - 8 × 256: ~1.5 GB
     - 10 × 200: ~1.5 GB
     - 12 × 200: ~1.8 GB

  3. 反向传播（只更新LoRA）:
     - 梯度: ~1.0 GB（固定，只有LoRA参数有梯度）
     - 临时中间值: ~0.5 GB

  4. 系统预留和碎片:
     - PyTorch预留: ~0.5 GB
     - CUDA碎片: ~0.5 GB

  动态占用总计:
    - 配置(8, 256):  2.5+1.5+1.0+0.5+1.0 = ~6.5 GB
    - 配置(10, 200): 2.4+1.5+1.0+0.5+1.0 = ~6.4 GB
    - 配置(12, 200): 2.9+1.8+1.0+0.5+1.0 = ~7.2 GB

总显存占用:
  配置(8, 256):  9.15 + 6.5  = ~15.65 GB ✅
  配置(10, 200): 9.15 + 6.4  = ~15.55 GB ✅
  配置(12, 200): 9.15 + 7.2  = ~16.35 GB ⚠️
```

### 配置对比

| 配置 | group_size | max_new_tokens | tokens/batch | 估算显存 | 风险等级 |
|------|-----------|----------------|--------------|----------|---------|
| 原始 | 8 | 256 | 2048 | 15.65 GB | 🟢 安全 |
| 方案A（推荐）| 10 | 200 | 2000 | 15.55 GB | 🟢 安全 |
| 方案B | 12 | 200 | 2400 | 16.35 GB | 🟡 可能OOM |
| 方案C（保守）| 8 | 180 | 1440 | 14.80 GB | 🟢 最安全 |

### 推荐配置：方案A

```yaml
group_size: 10
max_new_tokens: 200

优点:
  ✅ 显存安全（15.55 GB < 16 GB）
  ✅ 比原始配置(8,256)快约20%
  ✅ group_size更大，gradient估计更稳定
  ✅ 每step计算量: 2000 tokens（vs原始2048）

缺点:
  ⚠️ 比方案B的group_size=12略小
  ⚠️ 但差异不大（10 vs 12，仅20%）
```

---

## 5. 如果还是OOM怎么办？

### 诊断OOM来源

```python
# 如果训练时OOM，查看错误信息：

# 情况1: 前几步就OOM
→ 说明：模型加载或初始化时已经占满
→ 解决：降低group_size到8，或max_new_tokens到180

# 情况2: 训练一段时间后OOM
→ 说明：内存泄漏或某些step计算量特别大
→ 解决：检查是否有长文本prompt（>200 tokens）

# 情况3: 随机OOM（有时成功有时失败）
→ 说明：接近临界值，CUDA碎片导致
→ 解决：降低group_size或max_new_tokens
```

### 紧急降级方案

如果当前配置(10, 200)仍然OOM：

```bash
# 方案1：只降group_size
--group_size 8 \
--max_new_tokens 200 \

# 方案2：只降max_new_tokens
--group_size 10 \
--max_new_tokens 180 \

# 方案3：都降（最保守）
--group_size 8 \
--max_new_tokens 180 \
```

### 监控显存使用

训练开始后，在另一个终端运行：
```powershell
# 每5秒打印一次显存占用
while($true) { 
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader; 
    Start-Sleep -Seconds 5 
}
```

观察：
- 前10步的最大显存占用
- 如果接近16000 MB → 有OOM风险
- 如果稳定在15500 MB以下 → 安全

---

## 6. 最终配置建议

### 当前设置（已更新）

```yaml
reward配置:
  REWARD_BASELINE: 0.47 (可调)
  REWARD_SCALE_FACTOR: 5 (可调)

GRPO配置:
  group_size: 10 (安全)
  max_new_tokens: 200 (平衡)
  learning_rate: 2e-5 (大)
  warmup_steps: 15
  early_stopping_patience: 30

预估:
  显存: ~15.55 GB (安全)
  速度: ~5 min/prompt
  总时间: ~8 hours (100 prompts)
```

### 启动前检查清单

- [ ] 确认没有其他Python进程占用显存
- [ ] 确认USE_T5_PERTURBATION=1已设置
- [ ] 确认WandB token已配置
- [ ] 可选：设置REWARD_BASELINE（如果要调整）
- [ ] 可选：打开另一个终端监控显存

---

## 附录：相关代码位置

```yaml
Reward函数实现:
  文件: src/rl/reward_functions.py
  函数: detectgpt_pure_reward_with_info()
  行数: ~680-730

训练脚本:
  文件: src/rl/train_grpo.py
  
启动脚本:
  文件: start_sh/run_grpo_exp2.sh
  
相关文档:
  - docs/experiment2_reward_scaling.md (整体优化方案)
  - docs/gpu_optimization_log.md (GPU优化历史)
  - docs/experiment2_command.md (命令文档)
```
