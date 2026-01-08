"""
Reward 函数集合

这里的目标是：**用可解释、可复现的指标提升生成质量/可读性/“更像人类写作”**，从而做毕业设计的
reward 消融对比实验。

说明：我不会提供“针对某个检测器绕过/对抗”的奖励设计或实现；如果你需要做检测器相关评估，建议将
检测器作为**离线评测指标**而不是训练时的对抗奖励，以符合学术诚信与安全合规。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple


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
    """
    name = (name or "").strip().lower()
    if name in {"simple", "simple_reward"}:
        return simple_reward
    if name in {"humanlike_v1", "human_v1", "humanlike"}:
        return humanlike_reward_v1
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

    # 1) 长度：理想 120~260 左右（按字符近似），太短/太长都降
    # 用 bell-shaped：mu=200, sigma=90
    length_score = _bell(float(length), mu=200.0, sigma=90.0)  # 0~1

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
