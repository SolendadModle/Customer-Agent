"""
QianwenBot - 基于千问大模型的电商智能客服 Bot
整合 RAPTOR 树形检索、FAISS 向量检索、NLU、情感分析、多轮对话和智能推荐
"""

import os
from typing import Optional

from Agent.bot import Bot
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from config import config
from utils.logger import get_logger

from Agent.QianwenAgent.generation.qianwen_client import QianwenClient
from Agent.QianwenAgent.generation.rag_generator import RAGGenerator
from Agent.QianwenAgent.nlu.entity_extraction import NLUService
from Agent.QianwenAgent.sentiment.sentiment_analyzer import SentimentAnalyzer
from Agent.QianwenAgent.dialog.session_manager import SessionManager
from Agent.QianwenAgent.dialog.dialog_handler import DialogHandler
from Agent.QianwenAgent.recommendation.recommendation_engine import RecommendationEngine
from Agent.QianwenAgent.preprocessing import preprocessor


class QianwenBot(Bot):
    """
    千问智能客服 Bot

    处理流程：
    1. 消息预处理
    2. NLU 意图识别 + 实体抽取
    3. 情感分析（检测负面情绪，必要时转人工）
    4. RAPTOR / FAISS 知识库检索
    5. RAG + 千问大模型生成回复
    6. 智能推荐（附加推荐内容）
    7. 返回 Reply
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger("QianwenBot")

        # 读取配置
        api_key = config.get("dashscope_api_key") or os.environ.get("DASHSCOPE_API_KEY", "")
        model = config.get("qianwen_model", "qwen-turbo")
        kb_path = config.get("knowledge_base_path", "./data/knowledge_base")
        vector_store_path = config.get("vector_store_path", "./data/vector_store")
        raptor_enabled = config.get("raptor_enabled", True)
        sentiment_enabled = config.get("sentiment_enabled", True)
        recommendation_enabled = config.get("recommendation_enabled", True)
        nlu_enabled = config.get("nlu_enabled", True)

        # 初始化千问客户端
        self.qianwen_client = QianwenClient(api_key=api_key, model=model)

        # 初始化知识库检索（RAPTOR 或 FAISS）
        retrieval_service = self._init_retrieval(
            kb_path=kb_path,
            vector_store_path=vector_store_path,
            raptor_enabled=raptor_enabled,
        )

        # 初始化 RAG 生成器
        rag_generator = RAGGenerator(
            qianwen_client=self.qianwen_client,
            retrieval_service=retrieval_service,
        )

        # 初始化 NLU 服务
        self.nlu_service = NLUService() if nlu_enabled else None

        # 初始化情感分析器
        self.sentiment_analyzer = SentimentAnalyzer(enabled=sentiment_enabled)

        # 初始化多轮对话处理器
        session_manager = SessionManager()
        self.dialog_handler = DialogHandler(
            rag_generator=rag_generator,
            nlu_service=self.nlu_service or NLUService(),
            session_manager=session_manager,
        )

        # 初始化推荐引擎
        self.recommendation_engine = RecommendationEngine(enabled=recommendation_enabled)

        self.logger.info("QianwenBot 初始化完成")

    def _init_retrieval(self, kb_path: str, vector_store_path: str, raptor_enabled: bool):
        """初始化知识库检索服务"""
        try:
            if raptor_enabled:
                from Agent.QianwenAgent.raptor_retrieval import build_raptor_retriever
                from Agent.QianwenAgent.knowledge_base.kb_manager import KnowledgeBase
                kb = KnowledgeBase(kb_path=kb_path)
                retriever = build_raptor_retriever(
                    knowledge_base=kb,
                    qianwen_client=self.qianwen_client,
                    vector_store_path=vector_store_path,
                )
                self.logger.info("RAPTOR 树形检索服务已启动")
                return retriever
        except Exception as e:
            self.logger.warning(f"RAPTOR 初始化失败，回退到 FAISS 检索: {e}")

        try:
            from Agent.QianwenAgent.knowledge_base.kb_manager import KnowledgeBase
            from Agent.QianwenAgent.knowledge_base.retrieval import RetrievalService
            kb = KnowledgeBase(kb_path=kb_path)
            service = RetrievalService(
                knowledge_base=kb,
                vector_store_path=vector_store_path,
            )
            service.build_index()
            self.logger.info("FAISS 向量检索服务已启动")
            return service
        except Exception as e:
            self.logger.warning(f"FAISS 初始化失败，将不使用知识库检索: {e}")
            return None

    def reply(self, context: Context) -> Reply:
        """
        处理消息并返回回复

        Args:
            context: 消息上下文

        Returns:
            Reply 对象
        """
        try:
            shop_id = context.kwargs.get("shop_id", "")
            from_uid = context.kwargs.get("from_uid", "")
            # Ensure we always have a non-trivial user_id
            user_id = f"{shop_id}_{from_uid}" if (shop_id or from_uid) else "anonymous"

            # 1. 消息预处理
            raw_content = self._extract_text(context)
            if not raw_content:
                return Reply(ReplyType.TEXT, "您好，我收到了您的消息，请问有什么可以帮您？")

            query = preprocessor.preprocess(raw_content)

            # 2. 情感分析（高优先级，检测需要转人工的情况）
            should_escalate, escalation_reason = self.sentiment_analyzer.should_escalate(query)
            if should_escalate:
                self.logger.info(f"触发转人工: {escalation_reason}")
                return Reply(
                    ReplyType.TEXT,
                    f"非常抱歉给您带来了不便！我已将您的问题标记为紧急，"
                    f"客服专员将在最短时间内联系您，请稍候。"
                )

            # 3. 多轮对话处理（含 NLU + RAG 生成）
            result = self.dialog_handler.process_message(
                message=query,
                session_id=user_id,
                user_id=user_id,
                use_rag=True,
            )

            response_text = result.get("response", "抱歉，暂时无法回复，请稍后重试。")
            intent = result.get("intent", "general_question")
            entities = result.get("entities", {})

            # 4. 智能推荐（附加到回复末尾）
            recommendations = self.recommendation_engine.recommend_by_intent(intent, entities)
            if recommendations:
                rec_text = self._format_recommendations(recommendations)
                response_text = f"{response_text}\n\n{rec_text}"

            self.logger.info(f"[{user_id}] 意图: {intent} → 回复生成成功")
            return Reply(ReplyType.TEXT, response_text)

        except Exception as e:
            self.logger.error(f"处理消息异常: {e}", exc_info=True)
            return Reply(ReplyType.TEXT, "消息处理失败，请稍后重试或联系人工客服。")

    def _extract_text(self, context: Context) -> str:
        """从 Context 中提取文本内容"""
        if context.type == ContextType.TEXT:
            return str(context.content) if context.content else ""

        elif context.type in (ContextType.GOODS_INQUIRY, ContextType.GOODS_SPEC):
            try:
                goods_info = context.content
                if isinstance(goods_info, dict):
                    return (
                        f"商品咨询：{goods_info.get('goods_name', '')}，"
                        f"价格：{goods_info.get('goods_price', '')}，"
                        f"规格：{goods_info.get('goods_spec', '')}"
                    )
            except Exception:
                pass
            return "收到商品咨询"

        elif context.type == ContextType.ORDER_INFO:
            try:
                order_info = context.content
                if isinstance(order_info, dict):
                    return (
                        f"订单查询：订单号 {order_info.get('order_id', '')}，"
                        f"商品：{order_info.get('goods_name', '')}"
                    )
            except Exception:
                pass
            return "收到订单查询"

        elif context.type == ContextType.EMOTION:
            return str(context.content) if context.content else ""

        return ""

    def _format_recommendations(self, recommendations: list) -> str:
        """格式化推荐内容"""
        if not recommendations:
            return ""
        lines = ["\n💡 **相关推荐：**"]
        for item in recommendations[:3]:
            lines.append(f"• **{item['name']}**：{item['description']}")
        return "\n".join(lines)
