"""
Reward 函数集合（先保证流程能跑通）

后续你可以在这里替换成 DetectGPT / GPTZero / 组合奖励等更复杂的 reward。
"""

from __future__ import annotations

import re


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


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
