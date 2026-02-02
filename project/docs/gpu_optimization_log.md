# GPU优化日志 - 实验2性能优化

## 📊 优化前状态

| 指标 | 数值 |
|------|------|
| VRAM使用 | 16GB / 16GB (100%) |
| GPU利用率 | ~40% |
| 训练速度 | ~7-9小时 (100 prompts) |
| 瓶颈 | DetectGPT串行计算 |

### 问题分析

```python
# 优化前：串行计算（慢）
for _ in range(10):  # 10次扰动
    perturbed = perturb_text(text)
    logprob = compute_log_prob(perturbed)  # 逐个GPU计算
    
总计：8个候选 × 10次扰动 = 80次串行GPU调用/prompt
GPU大部分时间在等待 → 利用率只有40%
```

---

## 🚀 实施的优化

### 优化1：批处理计算（核心优化）

**修改位置：** `src/rl/reward_functions.py`

**新增方法：** `compute_log_prob_batch()`

```python
# 优化后：批量计算（快）
perturbed_texts = [perturb_text(text) for _ in range(10)]  # 先生成所有
logprobs = compute_log_prob_batch(perturbed_texts)  # 一次GPU计算！

批量forward：
- 10个文本 → 1个tensor (batch_size=10)
- GPU并行处理所有文本
- 减少CPU-GPU通信次数
```

**关键代码：**

```python
def compute_log_prob_batch(self, texts: list[str]) -> list[float]:
    # 批量tokenize（自动padding）
    inputs = self.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,  # 关键！
        truncation=True,
        max_length=512
    ).to(self.device)
    
    with torch.no_grad():
        # 一次forward计算所有文本
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        ...
```

**修改方法：** `detect()`

```python
# 修改前
for _ in range(self.num_perturbations):
    perturbed = self.perturb_text_t5(text)
    logprob = self.compute_log_prob(perturbed)  # 串行
    perturbed_logprobs.append(logprob)

# 修改后
perturbed_texts = [self.perturb_text_t5(text) 
                   for _ in range(self.num_perturbations)]
perturbed_logprobs = self.compute_log_prob_batch(perturbed_texts)  # 批量
```

---

### 优化2：混合精度计算

**原理：** 使用bfloat16而不是float32进行计算

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = self.model(**inputs, labels=inputs["input_ids"])
```

**优势：**
- 计算速度提升：~30%
- 显存占用减少：~50%
- 数值稳定性：bfloat16比fp16更稳定

---

## 📈 预期效果

| 指标 | 优化前 | 优化后 | 改善 |
|------|-------|-------|------|
| **GPU利用率** | 40% | 70-80% | ⬆️ **100%** |
| **训练时间** | 7-9h | 3-5h | ⬇️ **50-60%** |
| **每prompt耗时** | 4-5分钟 | 2-3分钟 | ⬇️ **50%** |
| **VRAM使用** | 16GB | 14-15GB | ⬇️ 10% |
| **准确性** | 100% | 100% | ✅ 不变 |

---

## 🔍 技术细节

### 为什么批处理提升这么大？

```
串行计算（优化前）：
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Text 1  │ Text 2  │ Text 3  │ Text 4  │ Text 5  │
└─────────┴─────────┴─────────┴─────────┴─────────┘
  1分钟      1分钟      1分钟      1分钟      1分钟
  总计：5分钟

批量计算（优化后）：
┌──────────────────────────────────────────────────┐
│ Batch [Text 1, 2, 3, 4, 5] - 并行处理            │
└──────────────────────────────────────────────────┘
  1.5分钟
  总计：1.5分钟

加速比：5 / 1.5 ≈ 3.3倍！
```

### 为什么GPU利用率低？

```
优化前的问题：
1. CPU生成扰动文本（慢）
   ↓
2. GPU计算1个文本的log_prob（快，但只用了10%的GPU）
   ↓
3. CPU收集结果
   ↓
4. 回到步骤1（重复80次）

GPU大部分时间在等待CPU → 利用率只有40%

优化后：
1. CPU批量生成10个扰动文本
   ↓
2. GPU一次计算10个文本（GPU满载！）
   ↓
3. CPU收集结果
   ↓
4. 重复次数减少8倍

GPU等待时间减少 → 利用率提升到70-80%
```

---

## 🎯 实验配置（优化版）

```yaml
# DetectGPT配置
num_perturbations: 10
eval_model: Qwen3-1.7B
use_t5_perturbation: True
batch_processing: True  # ✅ 新增
mixed_precision: True   # ✅ 新增

# GRPO配置
group_size: 8
max_new_tokens: 256
learning_rate: 5e-6
warmup_steps: 10
early_stopping_patience: 20
max_prompts: 100

# 性能优化
compute_log_prob_batch: True  # ✅ 批处理
autocast: bfloat16           # ✅ 混合精度
```

---

## 📝 代码修改摘要

### 文件：`src/rl/reward_functions.py`

**新增方法：**
1. `compute_log_prob_batch()` - 批量计算log概率（核心）

**修改方法：**
1. `detect()` - 使用批处理代替串行

**代码行数：**
- 新增：约80行
- 修改：约15行

---

## ⚠️ 注意事项

### 1. Padding的影响

批处理需要将不同长度的文本padding到相同长度：

```python
texts = ["Short text", "This is a much longer text..."]
# Padding后：
# ["Short text<pad><pad><pad>...", "This is a much longer text..."]
```

**处理方法：**
- 使用attention_mask区分真实token和padding
- 计算loss时只考虑非padding部分
- 不影响准确性

### 2. 混合精度的数值稳定性

bfloat16 vs float32：
- bfloat16范围大，适合深度学习
- 对于loss计算，精度足够
- 已在代码中处理异常情况

### 3. 批处理大小

当前批处理大小：10（num_perturbations）

如果想进一步提升：
- 可以增加批处理大小到20-50
- 需要修改`num_perturbations`
- 权衡准确性和速度

---

## 🚀 启动优化后的训练

```powershell
# 所有优化已自动启用
# 直接运行即可

conda activate biyeshejihuanjing

python biyesheji/project/src/rl/train_grpo.py \
  --base_model Qwen/Qwen3-7B \
  --sft_adapter_path biyesheji/project/models/sft/sft_20260105_1453 \
  --rl_data_path biyesheji/project/data/processed/rl_prompts.jsonl \
  --output_dir biyesheji/project/models/rl_detectgpt_pure_exp2 \
  --reward_name detectgpt_pure \
  --max_prompts 100 \
  --group_size 8 \
  --max_new_tokens 256 \
  --learning_rate 5e-6 \
  --warmup_steps 10 \
  --early_stopping_patience 20 \
  --report_to wandb \
  --wandb_project graduation_grpo \
  --run_name grpo_detectgpt_pure_t5_exp2_optimized_v2
```

---

## 📊 预期时间线

```
优化前估计：
- 每prompt：4-5分钟
- 100 prompts：7-9小时
- 完成时间：今晚7-9点

优化后估计：
- 每prompt：2-3分钟  ⬇️ 50%
- 100 prompts：3-5小时  ⬇️ 55%
- 完成时间：下午2-4点  🚀

如果early stopping触发：
- 可能在50-70步结束
- 完成时间：中午12-2点
```

---

## ✅ 优化验证清单

训练开始后，检查以下指标：

- [ ] GPU利用率：应该在70-80%（之前40%）
- [ ] 每prompt耗时：应该在2-3分钟（之前4-5分钟）
- [ ] VRAM使用：应该在14-15GB（之前16GB）
- [ ] WandB日志正常
- [ ] Loss和Reward趋势合理
- [ ] 没有OOM错误

---

## 🎓 技术总结

**本次优化的核心思想：**

> 从"逐个处理"到"批量处理"
> 从"等待时间"到"计算时间"
> 从"串行执行"到"并行计算"

**适用场景：**
- 任何需要重复相同计算的场景
- 多个独立样本的推理
- RL中的多候选评估

**推广价值：**
- 可应用于其他reward function
- 可应用于evaluation阶段
- 通用的GPU优化技巧

---

## 📞 问题排查

如果优化后出现问题：

### 问题1：OOM (内存不足)
```python
# 解决：减少批处理大小
num_perturbations: 10 → 5
```

### 问题2：精度问题
```python
# 解决：禁用混合精度
# 在compute_log_prob_batch()中设置：
use_amp = False
```

### 问题3：速度没提升
```bash
# 检查：
nvidia-smi  # 确认GPU被正确使用
# 查看日志中的批处理信息
```

---

## 📈 后续优化方向

如果还需要进一步提速：

1. **T5扰动批处理**（复杂度高）
   - 批量mask-filling
   - 预期提速：额外20-30%

2. **Model quantization**（量化）
   - 使用4-bit或8-bit量化
   - 预期提速：30-50%，但可能影响准确性

3. **分布式训练**（多GPU）
   - 需要多张GPU
   - 线性加速

4. **Compilation优化**（torch.compile）
   - PyTorch 2.0特性
   - 预期提速：10-20%

---

## 🎯 优化成功标志

✅ GPU利用率 >70%
✅ 训练时间 <5小时
✅ 准确性不变
✅ 稳定运行无错误

**预计效果：实验时间减半，GPU使用效率翻倍！** 🚀
