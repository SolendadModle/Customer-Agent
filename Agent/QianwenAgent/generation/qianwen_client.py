"""
千问大模型客户端
调用阿里云 DashScope API，支持单次生成和流式生成
"""

import os
from typing import Dict, List, Optional
import dashscope
from dashscope import Generation

from utils.logger import get_logger


class QianwenClient:
    """千问大模型 API 客户端"""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-turbo",
                 temperature: float = 0.7, top_p: float = 0.9,
                 max_tokens: int = 1000, repetition_penalty: float = 1.1):
        self.logger = get_logger("QianwenClient")
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.parameters = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
        }

        if self.api_key:
            dashscope.api_key = self.api_key
            self.logger.info(f"千问客户端初始化成功，模型: {self.model}")
        else:
            self.logger.warning("DASHSCOPE_API_KEY 未设置，千问 API 将无法使用")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None,
                 **kwargs) -> Optional[str]:
        """
        调用千问 API 生成回复

        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            history: 对话历史
            **kwargs: 额外参数（覆盖默认参数）

        Returns:
            生成的回复文本，失败时返回 None
        """
        if not self.api_key:
            self.logger.error("API 密钥未配置，无法调用千问 API")
            return None

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            params = {**self.parameters, **kwargs}

            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                max_tokens=params.get("max_tokens", 1000),
                repetition_penalty=params.get("repetition_penalty", 1.1),
            )

            if response.status_code == 200:
                result = response.output.choices[0].message.content
                self.logger.debug(f"生成回复: {result[:80]}...")
                return result
            else:
                self.logger.error(f"千问 API 错误: {response.code} - {response.message}")
                return None

        except Exception as e:
            self.logger.error(f"调用千问 API 异常: {e}")
            return None

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                        history: Optional[List[Dict[str, str]]] = None, **kwargs):
        """
        流式调用千问 API

        Yields:
            文本分块
        """
        if not self.api_key:
            self.logger.error("API 密钥未配置，无法调用千问 API（流式）")
            return

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})

            params = {**self.parameters, **kwargs}

            responses = Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                stream=True,
                incremental_output=True,
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                max_tokens=params.get("max_tokens", 1000),
                repetition_penalty=params.get("repetition_penalty", 1.1),
            )

            for response in responses:
                if response.status_code == 200:
                    chunk = response.output.choices[0].message.content
                    yield chunk
                else:
                    self.logger.error(f"流式生成错误: {response.code}")
                    break

        except Exception as e:
            self.logger.error(f"流式生成异常: {e}")
            yield "生成回复时发生错误，请稍后重试。"

    def check_api_status(self) -> bool:
        """检测 API 是否可用"""
        if not self.api_key:
            return False
        try:
            response = Generation.call(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"API 状态检测失败: {e}")
            return False
