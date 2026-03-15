"""
NLP 工具函数
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """清理文本：去除多余空白、特殊字符"""
    if not text:
        return ""
    # 去除多余空白
    text = re.sub(r"\s+", " ", text).strip()
    # 去除控制字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def split_sentences(text: str, max_length: int = 512) -> List[str]:
    """将文本按句子分割，并限制每段长度"""
    if not text:
        return []
    # 中英文句子分割
    segments = re.split(r"(?<=[。！？.!?])\s*", text)
    result = []
    current = ""
    for seg in segments:
        if len(current) + len(seg) <= max_length:
            current += seg
        else:
            if current:
                result.append(current.strip())
            current = seg
    if current:
        result.append(current.strip())
    return [s for s in result if s]


def truncate_text(text: str, max_length: int) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_conversation_history(history: List[dict]) -> str:
    """将对话历史格式化为字符串"""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "用户" if turn.get("role") == "user" else "客服"
        content = turn.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
