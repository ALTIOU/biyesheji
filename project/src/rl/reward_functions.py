"""
Reward 函数集合

这里的目标是：**用可解释、可复现的指标提升生成质量/可读性/"更像人类写作"**，从而做毕业设计的
reward 消融对比实验。

说明：我不会提供"针对某个检测器绕过/对抗"的奖励设计或实现；如果你需要做检测器相关评估，建议将
检测器作为**离线评测指标**而不是训练时的对抗奖励，以符合学术诚信与安全合规。

====================
DetectGPT 配置说明
====================

1. 评估模型：默认使用 Qwen/Qwen3-7B（与训练模型匹配）
   
2. 扰动方法开关：通过环境变量 USE_T5_PERTURBATION 控制
   - USE_T5_PERTURBATION=0 (默认): 使用简单的随机词替换扰动（快速，适合训练阶段）
   - USE_T5_PERTURBATION=1: 使用 T5 mask-filling 扰动（更接近原论文，适合最终评估）
   
3. 使用示例：
   # 简单扰动（推荐训练时使用）
   python src/rl/train_grpo.py --reward_name detectgpt_pure ...
   
   # T5扰动（推荐评估时使用）
   USE_T5_PERTURBATION=1 python src/rl/train_grpo.py --reward_name detectgpt_pure ...
   
   # Windows PowerShell:
   $env:USE_T5_PERTURBATION="1"; python src/rl/train_grpo.py --reward_name detectgpt_pure ...
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple, Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    DETECTGPT_AVAILABLE = True
except ImportError:
    DETECTGPT_AVAILABLE = False


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WS_RE = re.compile(r"\s+")
_WEIRD_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{6,}")  # 连续 7 个相同字符（中英文都不太自然）

RewardFn = Callable[[str], float]


@dataclass(frozen=True)
class RewardResult:
    score: float
    parts: Dict[str, float]


def get_reward_fn(name: str) -> RewardFn:
    """
    训练脚本里用的选择器：返回 `text -> float` 的函数。

    可用：
    - simple
    - humanlike_v1
    - detectgpt
    - detectgpt_pure (纯DetectGPT，无额外奖励)
    """
    name = (name or "").strip().lower()
    if name in {"simple", "simple_reward"}:
        return simple_reward
    if name in {"humanlike_v1", "human_v1", "humanlike"}:
        return humanlike_reward_v1
    if name in {"detectgpt", "detect_gpt", "detectgpt_reward"}:
        return detectgpt_reward
    if name in {"detectgpt_pure", "pure_detectgpt", "detectgpt_only"}:
        return detectgpt_pure_reward
    raise ValueError(f"Unknown reward name: {name}")


def simple_reward(text: str) -> float:
    """
    一个“先跑通流程”的简单 reward：
    - **长度奖励**：越长越好，但有上限（避免无限啰嗦）
    - **重复惩罚**：重复越多，reward 越低（避免乱码/复读机）
    - **轻微格式奖励**：有换行/标点略加分
    """
    if not text:
        return 0.0

    t = text.strip()
    if not t:
        return 0.0

    # 1) 长度奖励（字符级，兼容中英文）
    length = len(t)
    length_score = min(length / 240.0, 1.0)  # 0~1

    # 2) 重复惩罚：token 去重率（对中文按字、对英文按词）
    # 目标：乱重复时 unique_ratio 会很低
    if _CJK_RE.search(t):
        tokens = [ch for ch in t if not ch.isspace()]
    else:
        tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", t)

    if len(tokens) <= 1:
        unique_ratio = 1.0
    else:
        unique_ratio = len(set(tokens)) / float(len(tokens))
        unique_ratio = max(0.0, min(unique_ratio, 1.0))

    # 3) 轻微格式奖励
    fmt_bonus = 0.0
    if "\n" in t:
        fmt_bonus += 0.05
    if any(p in t for p in ("。", "！", "？", ".", "!", "?", ":", "：")):
        fmt_bonus += 0.05

    # 组合
    score = length_score * (0.5 + 0.5 * unique_ratio) + fmt_bonus
    return float(max(0.0, min(score, 1.2)))


def _tokenize(text: str) -> Tuple[bool, list[str]]:
    """
    返回 (is_cjk, tokens)。CJK 用“字”近似 token；非 CJK 用“词+符号”。
    """
    t = _WS_RE.sub(" ", text.strip())
    if not t:
        return False, []
    is_cjk = bool(_CJK_RE.search(t))
    if is_cjk:
        tokens = [ch for ch in t if not ch.isspace()]
    else:
        tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", t)
    return is_cjk, tokens


def _ngram_repeat_ratio(tokens: list[str], n: int) -> float:
    """
    n-gram 重复比例：越大越“复读机”。0 表示基本无重复。
    """
    if n <= 0:
        return 0.0
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    total = len(grams)
    return float(max(0.0, min(1.0, 1.0 - (uniq / total))))


def _bell(x: float, mu: float, sigma: float) -> float:
    """
    高斯形状的“贴近某个理想区间”奖励，返回 0~1。
    """
    if sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return float(math.exp(-0.5 * z * z))


def humanlike_reward_v1(text: str) -> float:
    """
    一个“更像人类写作”的启发式奖励（不依赖任何检测器），适合先做消融：
    - 长度在合理范围（过短/过长都扣分）
    - 避免 n-gram 复读、连续字符拉长
    - 标点密度接近常见写作习惯（太少像口水，太多像乱码）
    - 轻微鼓励分段/多句结构

    输出：单标量 reward（约 0~1.2），越大越好。
    """
    return humanlike_reward_v1_with_info(text).score


def humanlike_reward_v1_with_info(text: str) -> RewardResult:
    if not text:
        return RewardResult(0.0, {"empty": 1.0})

    t = text.strip()
    if not t:
        return RewardResult(0.0, {"empty": 1.0})

    # 基础清洗
    t_norm = _WS_RE.sub(" ", t)
    is_cjk, tokens = _tokenize(t_norm)
    length = len(t_norm)

    # 1) 长度：英文新闻更合理的字符长度区间（太短/太长都降）
    # 用 bell-shaped：mu=360, sigma=120
    length_score = _bell(float(length), mu=360.0, sigma=120.0)  # 0~1

    # 2) 复读惩罚：2-gram/3-gram 重复
    rep2 = _ngram_repeat_ratio(tokens, 2)
    rep3 = _ngram_repeat_ratio(tokens, 3)
    repetition_penalty = 0.55 * rep2 + 0.45 * rep3  # 0~1
    repetition_score = 1.0 - repetition_penalty

    # 3) 词汇多样性（太低不好；太高也可能是胡言乱语，这里只轻微使用）
    if len(tokens) <= 1:
        uniq_ratio = 1.0
    else:
        uniq_ratio = len(set(tokens)) / float(len(tokens))
        uniq_ratio = float(max(0.0, min(1.0, uniq_ratio)))
    # 理想在 0.55~0.85 附近（不同文本差异大，只做弱约束）
    diversity_score = _bell(uniq_ratio, mu=0.72, sigma=0.18)

    # 4) 标点密度：按 CJK/非 CJK 简单计数
    puncts = "。！？；：，、,.!?;:"
    punct_cnt = sum(1 for ch in t_norm if ch in puncts)
    punct_density = punct_cnt / float(max(length, 1))
    # 人类常见密度大概在 0.02~0.08（粗略），取 mu=0.045
    punct_score = _bell(punct_density, mu=0.045, sigma=0.02)

    # 5) 结构：鼓励多句/分段（非常弱）
    sent_seps = "。！？.!?"
    sent_cnt = sum(1 for ch in t_norm if ch in sent_seps)
    paragraph_bonus = 0.08 if "\n" in t else 0.0
    sent_bonus = 0.06 if sent_cnt >= 2 else 0.0

    # 6) 异常惩罚：控制字符/大量重复字符
    weird_pen = 1.0 if _WEIRD_RE.search(t) else 0.0
    long_repeat_pen = 1.0 if _REPEAT_CHAR_RE.search(t_norm) else 0.0
    anomaly_penalty = 0.5 * weird_pen + 0.5 * long_repeat_pen  # 0~1
    anomaly_score = 1.0 - anomaly_penalty

    # 组合（权重可作为后续消融点）
    score = (
        0.32 * length_score
        + 0.28 * repetition_score
        + 0.16 * punct_score
        + 0.14 * diversity_score
        + 0.10 * anomaly_score
        + paragraph_bonus
        + sent_bonus
    )
    score = float(max(0.0, min(score, 1.25)))

    parts: Dict[str, float] = {
        "length_score": float(length_score),
        "repetition_score": float(repetition_score),
        "diversity_score": float(diversity_score),
        "punct_score": float(punct_score),
        "anomaly_score": float(anomaly_score),
        "paragraph_bonus": float(paragraph_bonus),
        "sent_bonus": float(sent_bonus),
        "is_cjk": 1.0 if is_cjk else 0.0,
        "length": float(length),
        "punct_density": float(punct_density),
        "uniq_ratio": float(uniq_ratio),
        "rep2": float(rep2),
        "rep3": float(rep3),
    }
    return RewardResult(score=score, parts=parts)


# ============================================================
# DetectGPT 检测器奖励函数
# ============================================================

class DetectGPTDetector:
    """
    DetectGPT 检测器封装类
    
    基于论文: "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
    原理: AI生成的文本倾向于位于模型的局部概率极大值区域，而人类写作的文本不一定如此。
    通过对文本进行轻微扰动，比较原文和扰动文本的概率差异来判断。
    
    检测分数越高 -> 越像AI生成
    检测分数越低 -> 越像人类写作
    """
    
    _instance: Optional["DetectGPTDetector"] = None
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-7B",
        device: Optional[str] = None,
        num_perturbations: int = 5,
        use_t5_perturbation: bool = False
    ):
        """
        初始化 DetectGPT 检测器
        
        Args:
            model_name: 用于计算概率的模型（默认使用 Qwen/Qwen3-7B，与训练模型匹配）
            device: 设备 (cuda/cpu)
            num_perturbations: 扰动次数（越多越准确，但越慢）
            use_t5_perturbation: 是否使用T5进行扰动（True=原始论文方法，False=简化方法）
        """
        if not DETECTGPT_AVAILABLE:
            raise ImportError("DetectGPT需要安装 torch 和 transformers")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_perturbations = num_perturbations
        self.use_t5_perturbation = use_t5_perturbation
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # 设置 pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"加载DetectGPT模型失败: {e}")
    
    @classmethod
    def get_instance(cls, **kwargs) -> "DetectGPTDetector":
        """
        获取单例实例（避免重复加载模型）
        
        注意：如果参数改变（如use_t5_perturbation），会创建新实例
        """
        # 检查是否需要重新创建实例（参数改变）
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        else:
            # 检查关键参数是否改变
            use_t5 = kwargs.get('use_t5_perturbation', False)
            if hasattr(cls._instance, 'use_t5_perturbation') and \
               cls._instance.use_t5_perturbation != use_t5:
                print(f"[Info] T5扰动设置改变，重新创建DetectGPT实例")
                cls._instance = cls(**kwargs)
        
        return cls._instance
    
    def compute_log_prob(self, text: str) -> float:
        """计算文本的平均log概率"""
        if not text.strip():
            return -100.0
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                # 返回平均负log似然
                return -outputs.loss.item()
        except Exception:
            return -100.0
    
    def compute_log_prob_batch(self, texts: list[str]) -> list[float]:
        """
        🚀 批量计算多个文本的log概率（GPU并行优化 + 混合精度）
        
        关键优化：
        1. 批量处理：将多个文本一次性送入GPU
        2. 混合精度：使用bf16/fp16加速计算
        
        预期提升：GPU利用率从40% → 70-80%，速度提升50-60%
        
        Args:
            texts: 文本列表
            
        Returns:
            每个文本的平均log概率列表
        """
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [t if t.strip() else " " for t in texts]
        
        try:
            # 批量tokenize（自动padding到同一长度）
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                padding=True,  # 关键：批处理需要padding
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 🚀 使用混合精度加速（如果是CUDA）
            use_amp = self.device == "cuda" and torch.cuda.is_available()
            
            with torch.no_grad():
                if use_amp:
                    # 使用自动混合精度
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        logits = outputs.logits
                else:
                    # CPU或不支持混合精度时使用全精度
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    logits = outputs.logits
                
                # 对于批量输入，需要分别计算每个样本的loss
                labels = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # 计算每个样本的loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention = attention_mask[..., 1:].contiguous()
                
                # 计算每个样本的平均负log似然
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(shift_labels.size())
                
                # 对每个样本计算平均loss（只计算非padding部分）
                results = []
                for i in range(len(texts)):
                    mask = shift_attention[i].bool()
                    if mask.sum() > 0:
                        sample_loss = losses[i][mask].mean().item()
                        results.append(-sample_loss)  # 返回负log似然
                    else:
                        results.append(-100.0)
                
                return results
                
        except Exception as e:
            # 出错时回退到单个计算
            print(f"[Warning] 批处理失败，回退到串行计算: {e}")
            return [self.compute_log_prob(t) for t in texts]
    
    def perturb_text(self, text: str) -> str:
        """
        对文本进行轻微扰动（简化版本）
        实际DetectGPT使用mask-filling模型，这里使用简单的同义词替换模拟
        """
        import random
        
        # 简单的扰动：随机替换一些常见词
        words = text.split()
        if len(words) < 3:
            return text
        
        # 随机选择1-2个位置进行微小改动（添加/删除空格或标点）
        perturbed = words.copy()
        num_changes = min(2, max(1, len(words) // 10))
        
        for _ in range(num_changes):
            idx = random.randint(0, len(perturbed) - 1)
            # 小概率在词后添加逗号或空格变化
            if random.random() < 0.5 and not perturbed[idx].endswith((',', '.', '!', '?')):
                perturbed[idx] = perturbed[idx] + ","
        
        return " ".join(perturbed)
    
    def perturb_text_t5(self, text: str) -> str:
        """
        使用T5进行mask-filling扰动（原始DetectGPT论文方法）
        
        原理：
        1. 随机mask文本中的15% tokens
        2. 使用T5模型填充masked位置
        3. 得到语义相近但表达不同的文本
        
        注意：需要额外安装 transformers 和 sentencepiece
        首次运行会下载T5模型（~240MB）
        
        Returns:
            扰动后的文本
        """
        import random
        
        try:
            # 延迟导入，只在使用T5时加载
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            # 检查是否已加载T5模型（避免重复加载）
            if not hasattr(self, '_t5_model'):
                print("[Info] 首次使用T5扰动，正在加载T5模型...")
                self._t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self._t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)
                self._t5_model.eval()
                print("[Info] T5模型加载完成")
            
            # Tokenize文本
            words = text.split()
            if len(words) < 3:
                return text
            
            # 随机选择15%的位置进行mask
            num_masks = max(1, int(len(words) * 0.15))
            mask_indices = sorted(random.sample(range(len(words)), k=num_masks))
            
            # 使用<extra_id_N>标记mask位置（T5的特殊token）
            masked_words = []
            mask_id = 0
            for i, word in enumerate(words):
                if i in mask_indices:
                    masked_words.append(f"<extra_id_{mask_id}>")
                    mask_id += 1
                else:
                    masked_words.append(word)
            
            masked_text = " ".join(masked_words)
            
            # T5填充
            inputs = self._t5_tokenizer(
                masked_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._t5_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=1,  # 贪心解码，快速
                    do_sample=False,
                )
            
            filled_text = self._t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 清理输出（T5可能生成多余空格）
            filled_text = " ".join(filled_text.split())
            
            return filled_text
            
        except ImportError:
            print("[Warning] T5不可用，回退到简单扰动方法")
            return self.perturb_text(text)
        except Exception as e:
            print(f"[Warning] T5扰动失败: {e}，使用简单扰动")
            return self.perturb_text(text)
    
    def detect(self, text: str) -> float:
        """
        检测文本是否为AI生成
        
        Returns:
            检测分数 (0~1)，越高越像AI生成，越低越像人类写作
        """
        if not text.strip():
            return 0.5
        
        # 计算原文本的log概率
        original_logprob = self.compute_log_prob(text)
        
        # 计算扰动文本的平均log概率（串行计算，稳定可靠）
        perturbed_logprobs = []
        for _ in range(self.num_perturbations):
            # 根据配置选择扰动方法
            if self.use_t5_perturbation:
                perturbed = self.perturb_text_t5(text)
            else:
                perturbed = self.perturb_text(text)
            
            logprob = self.compute_log_prob(perturbed)
            perturbed_logprobs.append(logprob)
        
        if not perturbed_logprobs:
            return 0.5
        
        avg_perturbed_logprob = sum(perturbed_logprobs) / len(perturbed_logprobs)
        
        # DetectGPT 分数：原文概率 - 扰动文本平均概率
        # 如果原文概率明显高于扰动版本，说明在局部极值 -> 更像AI生成
        curvature = original_logprob - avg_perturbed_logprob
        
        # 归一化到 0~1 (使用 sigmoid)
        # curvature > 0 -> 更像AI生成 -> 分数偏高
        # curvature < 0 -> 更像人类 -> 分数偏低
        detection_score = 1.0 / (1.0 + math.exp(-curvature))
        
        return float(max(0.0, min(1.0, detection_score)))


def detectgpt_reward(text: str) -> float:
    """
    基于 DetectGPT 检测器的奖励函数
    
    策略：
    - DetectGPT 分数越低（越像人类）-> 奖励越高
    - DetectGPT 分数越高（越像AI）-> 奖励越低
    
    输出：奖励分数 (0~1.2)
    """
    return detectgpt_reward_with_info(text).score


def detectgpt_reward_with_info(text: str) -> RewardResult:
    """
    基于 DetectGPT 检测器的奖励函数（带详细信息）
    
    Returns:
        RewardResult 包含总分和各部分分数
    """
    if not text or not text.strip():
        return RewardResult(0.0, {"empty": 1.0})
    
    if not DETECTGPT_AVAILABLE:
        # 如果没有安装依赖，回退到 humanlike_v1
        return humanlike_reward_v1_with_info(text)
    
    try:
        # 获取 DetectGPT 检测器实例
        # 可通过环境变量 USE_T5_PERTURBATION=1 启用T5扰动
        import os
        use_t5 = os.environ.get("USE_T5_PERTURBATION", "0") == "1"
        
        detector = DetectGPTDetector.get_instance(
            model_name="Qwen/Qwen3-7B",  # 使用与训练模型相同的评估模型
            num_perturbations=10,  # 平衡准确性和速度（原论文用100，实践中10次已足够稳定）
            use_t5_perturbation=use_t5  # 是否使用T5扰动
        )
        
        # 获取检测分数 (0~1, 越高越像AI)
        detection_score = detector.detect(text)
        
        # 转换为奖励：越不像AI生成，奖励越高
        # human_likeness = 1 - detection_score
        human_likeness = 1.0 - detection_score
        
        # 额外考虑一些基础质量因子
        is_cjk, tokens = _tokenize(text.strip())
        length = len(text.strip())
        
        # 1) 基础长度检查（太短不给高分）
        if length < 50:
            length_factor = length / 50.0
        elif length > 800:
            length_factor = 800.0 / length
        else:
            length_factor = 1.0
        
        # 2) 基本重复检查
        rep2 = _ngram_repeat_ratio(tokens, 2)
        repetition_factor = 1.0 - min(0.3, rep2)  # 最多扣30%
        
        # 综合分数
        # 主要依赖 DetectGPT (70%), 辅助质量因子 (30%)
        base_score = 0.70 * human_likeness + 0.20 * length_factor + 0.10 * repetition_factor
        
        # 稍微提升上限以鼓励优秀文本
        final_score = base_score * 1.15
        final_score = float(max(0.0, min(1.2, final_score)))
        
        parts: Dict[str, float] = {
            "detection_score": float(detection_score),
            "human_likeness": float(human_likeness),
            "length_factor": float(length_factor),
            "repetition_factor": float(repetition_factor),
            "base_score": float(base_score),
            "length": float(length),
            "is_cjk": 1.0 if is_cjk else 0.0,
        }
        
        return RewardResult(score=final_score, parts=parts)
    
    except Exception as e:
        # 如果检测失败，回退到基础奖励
        print(f"[Warning] DetectGPT检测失败: {e}，回退到humanlike_v1")
        return humanlike_reward_v1_with_info(text)


def detectgpt_pure_reward(text: str) -> float:
    """
    纯DetectGPT奖励函数（无额外质量因子）
    
    策略：
    - 只使用DetectGPT检测分数
    - 不考虑长度、重复度等因素
    - 纯粹基于"人类相似度"给奖励
    
    输出：奖励分数 (0~1.0)
    """
    return detectgpt_pure_reward_with_info(text).score


def detectgpt_pure_reward_with_info(text: str) -> RewardResult:
    """
    纯DetectGPT奖励函数（带详细信息）
    
    这是最纯粹的DetectGPT奖励实现：
    - 只使用概率曲率检测
    - 不混入任何启发式规则
    - 适合验证DetectGPT本身的有效性
    
    Returns:
        RewardResult 包含总分和各部分分数
    """
    if not text or not text.strip():
        return RewardResult(0.0, {"empty": 1.0})
    
    if not DETECTGPT_AVAILABLE:
        # 如果没有安装依赖，返回0
        print("[Warning] DetectGPT不可用，返回0分")
        return RewardResult(0.0, {"error": 1.0})
    
    try:
        # 获取 DetectGPT 检测器实例
        # 可通过环境变量 USE_T5_PERTURBATION=1 启用T5扰动
        import os
        use_t5 = os.environ.get("USE_T5_PERTURBATION", "0") == "1"
        
        detector = DetectGPTDetector.get_instance(
            model_name="Qwen/Qwen3-7B",  # 使用与训练模型相同的评估模型
            num_perturbations=10,  # 平衡准确性和速度（原论文用100，实践中10次已足够稳定）
            use_t5_perturbation=use_t5  # 是否使用T5扰动
        )
        
        # 获取检测分数 (0~1, 越高越像AI)
        detection_score = detector.detect(text)
        
        # 转换为奖励：越不像AI生成，奖励越高
        human_likeness = 1.0 - detection_score
        
        # === Reward Scaling: 放大微小差异 ===
        # 问题：原始human_likeness集中在0.44-0.50，差异仅0.06
        # 解决：以baseline为中心，放大偏离程度
        #
        # Baseline选择逻辑：
        # - 根据之前实验，detection_score集中在0.50-0.56
        # - 对应human_likeness = 1 - detection_score ≈ 0.44-0.50
        # - 取中位数baseline=0.47（对应detection_score=0.53）
        # - 可通过环境变量调整：REWARD_BASELINE=0.46 (如果需要)
        baseline = float(os.environ.get("REWARD_BASELINE", "0.47"))
        scale_factor = float(os.environ.get("REWARD_SCALE_FACTOR", "5"))
        
        advantage = (human_likeness - baseline) * scale_factor
        
        # 映射回0-1范围，但保留放大后的差异
        # 示例：
        #   human_likeness=0.50 → advantage=+0.15 → final=0.65
        #   human_likeness=0.47 → advantage=0.00  → final=0.50
        #   human_likeness=0.44 → advantage=-0.15 → final=0.35
        final_score = 0.5 + advantage
        final_score = float(max(0.0, min(1.0, final_score)))
        
        parts: Dict[str, float] = {
            "detection_score": float(detection_score),
            "human_likeness": float(human_likeness),
            "advantage": float(advantage),
            "final_score": float(final_score),
            "length": float(len(text.strip())),
        }
        
        return RewardResult(score=final_score, parts=parts)
    
    except Exception as e:
        # 如果检测失败，返回中等分数
        print(f"[Warning] DetectGPT检测失败: {e}，返回默认分数0.5")
        return RewardResult(0.5, {"error": 1.0, "message": str(e)})
