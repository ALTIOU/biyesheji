# SFT 训练环境配置说明

## 环境配置状态 ✅

你的 `biyesheji` conda 环境已经配置完成，包含以下包：

- **PyTorch 2.9.1** (CPU 版本)
- **transformers 4.57.3**
- **datasets 2.21.0**
- **peft 0.18.0**
- **accelerate 1.12.0**
- **bitsandbytes 0.49.0**

## 训练脚本配置

已优化适用于 macOS CPU 环境：
- 移除了 4bit 量化（macOS 不支持）
- 调整了 LoRA 参数（r=8, alpha=16）
- 设置了 CPU 设备映射
- 减少了训练 epoch 到 1（用于测试）
- 设置 `dataloader_num_workers=0`

## 运行训练

### 方法 1：使用启动脚本（推荐）
```bash
./run_sft_training.sh
```

### 方法 2：手动运行
```bash
# 激活环境
conda activate biyesheji

# 设置环境变量解决 OpenMP 冲突
export KMP_DUPLICATE_LIB_OK=TRUE

# 运行训练
python project/src/sft/train_sft_lora.py
```

## 训练配置

- **基础模型**: Qwen/Qwen3-1.7B
- **LoRA 配置**: r=8, alpha=16, target_modules=["q_proj","v_proj"]
- **训练数据**: 3000 条 SFT 数据
- **批次大小**: 1 (gradient_accumulation=4)
- **训练轮数**: 1 (epoch)
- **学习率**: 2e-4

## 预期输出

训练完成后将在 `project/models/sft/` 目录生成：
- `adapter_model.bin` - LoRA 适配器权重
- `adapter_config.json` - LoRA 配置
- `tokenizer/` - 分词器文件

## 注意事项

1. **首次运行**: 模型下载可能需要较长时间（1.7B 模型约 3GB）
2. **内存使用**: CPU 训练需要约 8-16GB RAM
3. **训练时间**: 在 M1/M2 Mac 上约需要 30-60 分钟完成 1 个 epoch
4. **OpenMP 警告**: macOS 上的正常现象，已通过环境变量解决

## 测试环境

运行以下命令验证环境：
```bash
conda run -n biyesheji python test_sft_env.py
```

## 下一步

训练完成后，你可以：
1. 使用训练好的模型进行推理测试
2. 调整训练参数进行更长时间的训练
3. 开始 RL 训练阶段
