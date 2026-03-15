"""
RAG 生成器
结合知识库检索与千问大模型，实现检索增强生成
"""

from typing import Dict, List, Optional

from Agent.QianwenAgent.generation.qianwen_client import QianwenClient
from utils.logger import get_logger


# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的电商平台智能客服助手，熟悉商品、订单、退款、物流等相关业务，"
    "请根据提供的知识库内容，准确、礼貌地回答用户问题。"
)

# 默认 RAG 提示词模板
DEFAULT_RAG_TEMPLATE = (
    "知识库内容：\n{context}\n\n"
    "对话历史：\n{history}\n\n"
    "用户问题：{question}\n\n"
    "请根据以上知识库内容和对话历史，给出专业、友好的回复："
)


def _format_history(history: List[Dict[str, str]]) -> str:
    """将对话历史格式化为字符串"""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = "用户" if turn.get("role") == "user" else "客服"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


class RAGGenerator:
    """检索增强生成器"""

    def __init__(self, qianwen_client: Optional[QianwenClient] = None,
                 retrieval_service=None,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 rag_template: str = DEFAULT_RAG_TEMPLATE):
        self.logger = get_logger("RAGGenerator")
        self.qianwen_client = qianwen_client or QianwenClient()
        self.retrieval_service = retrieval_service  # 可注入 RetrievalService 或 RaptorRetriever
        self.system_prompt = system_prompt
        self.rag_template = rag_template

    def generate_response(self, query: str,
                          conversation_history: Optional[List[Dict[str, str]]] = None,
                          use_rag: bool = True, **kwargs) -> str:
        """
        生成回复

        Args:
            query: 用户问题
            conversation_history: 对话历史
            use_rag: 是否启用 RAG 检索
            **kwargs: 额外参数

        Returns:
            生成的回复文本
        """
        try:
            history_str = _format_history(conversation_history or [])

            if use_rag and self.retrieval_service:
                context = self.retrieval_service.retrieve_context(query)
            else:
                context = "暂无相关知识库内容。"

            prompt = self.rag_template.format(
                context=context,
                history=history_str,
                question=query,
            )

            response = self.qianwen_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                **kwargs,
            )

            if response:
                self.logger.info(f"RAG 生成回复成功，查询: {query[:40]}...")
                return response
            return "抱歉，暂时无法生成回复，请稍后重试。"

        except Exception as e:
            self.logger.error(f"RAG 生成异常: {e}")
            return "处理您的问题时发生错误，请稍后重试。"

    def generate_with_context(self, query: str, context: str,
                              conversation_history: Optional[List[Dict[str, str]]] = None,
                              **kwargs) -> str:
        """使用外部传入的 context 生成回复（不触发内部检索）"""
        try:
            history_str = _format_history(conversation_history or [])
            prompt = self.rag_template.format(
                context=context,
                history=history_str,
                question=query,
            )
            response = self.qianwen_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                **kwargs,
            )
            return response or "抱歉，暂时无法生成回复，请稍后重试。"
        except Exception as e:
            self.logger.error(f"带上下文生成异常: {e}")
            return "处理您的问题时发生错误，请稍后重试。"
