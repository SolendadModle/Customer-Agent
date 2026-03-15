"""
文本预处理模块
对用户输入进行清理、规范化，提高 NLU 和检索质量
"""

import re
from typing import Optional

from Agent.QianwenAgent.nlp_utils import clean_text


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self, max_length: int = 1000):
        self.max_length = max_length

    def preprocess(self, text: str) -> str:
        """
        预处理流水线：清理 → 规范化 → 截断

        Args:
            text: 原始文本

        Returns:
            处理后的文本
        """
        if not text:
            return ""

        text = clean_text(text)
        text = self._normalize(text)
        text = self._truncate(text)
        return text

    def _normalize(self, text: str) -> str:
        """规范化：全角转半角、URL 替换"""
        # 全角数字/字母转半角
        result = []
        for ch in text:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif ch == "\u3000":
                result.append(" ")
            else:
                result.append(ch)
        text = "".join(result)
        # 替换 URL
        text = re.sub(r"https?://\S+", "[链接]", text)
        return text

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_length:
            return text[: self.max_length] + "..."
        return text


# 全局预处理器实例
preprocessor = TextPreprocessor()
