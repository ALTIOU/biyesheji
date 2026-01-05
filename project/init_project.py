import os

# =============================
# 配置：你的项目主目录名称
# =============================
PROJECT_ROOT = "project"

# =============================
# 要创建的全部目录结构
# =============================
DIRS = [
    f"{PROJECT_ROOT}/data/raw",
    f"{PROJECT_ROOT}/data/processed",
    f"{PROJECT_ROOT}/data/prompts",

    f"{PROJECT_ROOT}/models/base",
    f"{PROJECT_ROOT}/models/sft",
    f"{PROJECT_ROOT}/models/rl",
    f"{PROJECT_ROOT}/models/tokenizer",

    f"{PROJECT_ROOT}/detectors/detectgpt",
    f"{PROJECT_ROOT}/detectors/gptzero",
    f"{PROJECT_ROOT}/detectors/radar",

    f"{PROJECT_ROOT}/src/data_preprocess",
    f"{PROJECT_ROOT}/src/sft",
    f"{PROJECT_ROOT}/src/rl",
    f"{PROJECT_ROOT}/src/evaluate",
    f"{PROJECT_ROOT}/src/utils",

    f"{PROJECT_ROOT}/configs",

    f"{PROJECT_ROOT}/outputs/sft",
    f"{PROJECT_ROOT}/outputs/rl",
    f"{PROJECT_ROOT}/outputs/detect_results",
    f"{PROJECT_ROOT}/outputs/logs",

    f"{PROJECT_ROOT}/notebooks",
]

# =============================
# 要创建的占位文件
# =============================
FILES = {
    f"{PROJECT_ROOT}/README.md":
"""# 毕业设计项目说明

该目录包含代码、数据、模型及实验结果。
""",

    f"{PROJECT_ROOT}/configs/sft_config.yaml": "# SFT 配置文件\n",
    f"{PROJECT_ROOT}/configs/rl_config.yaml": "# RL (PPO/GRPO) 配置文件\n",
    f"{PROJECT_ROOT}/configs/eval_config.yaml": "# 检测器评估配置文件\n",

    f"{PROJECT_ROOT}/src/data_preprocess/__init__.py": "",
    f"{PROJECT_ROOT}/src/sft/__init__.py": "",
    f"{PROJECT_ROOT}/src/rl/__init__.py": "",
    f"{PROJECT_ROOT}/src/evaluate/__init__.py": "",
    f"{PROJECT_ROOT}/src/utils/__init__.py": "",

    f"{PROJECT_ROOT}/src/data_preprocess/prepare_dataset.py":
"# 数据集下载与预处理脚本（待填写）\n",

    f"{PROJECT_ROOT}/src/sft/train_sft_lora.py":
"# LoRA SFT 训练脚本（待填写）\n",

    f"{PROJECT_ROOT}/src/rl/train_ppo.py":
"# PPO 训练脚本（待填写）\n",

    f"{PROJECT_ROOT}/src/rl/reward_functions.py":
"# 奖励函数定义（待填写）\n",

    f"{PROJECT_ROOT}/src/evaluate/evaluate_detectgpt.py":
"# DetectGPT 评估脚本（待填写）\n",

    f"{PROJECT_ROOT}/src/evaluate/evaluate_gptzero.py":
"# GPTZero 评估脚本（待填写）\n",

    f"{PROJECT_ROOT}/notebooks/analysis.ipynb": "",
}

# =============================
# 执行目录和文件创建
# =============================
print("🔧 正在初始化项目结构...\n")

for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"📁 创建目录：{d}")

print("\n📝 正在创建占位文件...\n")

for file_path, content in FILES.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 创建文件：{file_path}")

print("\n🎉 项目结构创建完成！结构如下：\n")

for d in DIRS:
    print(" -", d)

print("\n你现在可以开始填代码了！🚀")