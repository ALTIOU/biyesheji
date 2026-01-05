#!/bin/bash
# SFT 训练启动脚本

# 设置环境变量解决 macOS OpenMP 冲突
export KMP_DUPLICATE_LIB_OK=TRUE

# 激活环境
conda activate biyesheji

# 运行训练脚本
cd /Users/altiou/code_learn/graduation_project
python project/src/sft/train_sft_lora.py
