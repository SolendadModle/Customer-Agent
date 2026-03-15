"""
多轮对话处理器
整合 NLU、情感分析、RAG 生成，管理多轮对话流程
"""

from typing import Any, Dict, List, Optional

from Agent.QianwenAgent.dialog.session_manager import SessionManager, ConversationSession
from Agent.QianwenAgent.generation.rag_generator import RAGGenerator
from Agent.QianwenAgent.nlu.entity_extraction import NLUService
from utils.logger import get_logger


# 意图 → 会话状态映射
INTENT_STATE_MAP = {
    "greeting": "initial",
    "farewell": "closed",
    "product_inquiry": "information_gathering",
    "order_status": "information_gathering",
    "return_request": "information_gathering",
    "shipping_inquiry": "information_gathering",
    "payment_issue": "information_gathering",
    "complaint": "escalation",
}


class DialogHandler:
    """多轮对话处理器"""

    def __init__(self, rag_generator: Optional[RAGGenerator] = None,
                 nlu_service: Optional[NLUService] = None,
                 session_manager: Optional[SessionManager] = None,
                 context_window: int = 5):
        self.logger = get_logger("DialogHandler")
        self.rag_generator = rag_generator or RAGGenerator()
        self.nlu_service = nlu_service or NLUService()
        self.session_manager = session_manager or SessionManager()
        self.context_window = context_window
        self.logger.info("多轮对话处理器初始化完成")

    def process_message(self, message: str,
                        session_id: Optional[str] = None,
                        user_id: Optional[str] = None,
                        use_rag: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """
        处理用户消息，返回回复及元数据

        Args:
            message: 用户消息
            session_id: 会话 ID（可选）
            user_id: 用户 ID（可选）
            use_rag: 是否启用 RAG 检索
            **kwargs: 额外参数传递给 RAGGenerator

        Returns:
            {session_id, response, intent, intent_confidence, entities, state, turn_count}
        """
        try:
            session = self.session_manager.get_or_create_session(session_id, user_id)

            # NLU 分析
            nlu_result = self.nlu_service.analyze(message)
            intent = nlu_result["intent"]["name"]
            intent_confidence = nlu_result["intent"]["confidence"]
            entities = nlu_result["entities"]

            # 记录用户消息
            session.add_message("user", message, metadata={
                "intent": intent,
                "intent_confidence": intent_confidence,
                "entities": entities,
            })

            # 更新实体上下文
            for entity_type, entity_list in entities.items():
                if entity_list:
                    session.update_context(f"last_{entity_type.lower()}", entity_list[0])

            # 更新会话状态
            new_state = INTENT_STATE_MAP.get(intent, session.state)
            session.set_state(new_state)

            # 获取历史（不含当前消息）
            history = session.get_history(max_turns=self.context_window)
            formatted_history = [
                {"role": t["role"], "content": t["content"]}
                for t in history[:-1]
            ]

            # 生成回复
            response = self.rag_generator.generate_response(
                query=message,
                conversation_history=formatted_history,
                use_rag=use_rag,
                **kwargs,
            )

            # 记录助手回复
            session.add_message("assistant", response, metadata={
                "intent": intent,
                "state": session.state,
            })

            return {
                "session_id": session.session_id,
                "response": response,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "entities": entities,
                "state": session.state,
                "turn_count": len(session.history) // 2,
            }

        except Exception as e:
            self.logger.error(f"处理消息异常: {e}", exc_info=True)
            return {
                "session_id": session_id,
                "response": "抱歉，处理您的问题时发生了错误，请稍后重试。",
                "error": str(e),
            }

    def end_session(self, session_id: str):
        """结束会话"""
        self.session_manager.delete_session(session_id)
        self.logger.info(f"会话已结束: {session_id}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        session = self.session_manager.get_session(session_id)
        return session.to_dict() if session else None

    def get_statistics(self) -> Dict[str, Any]:
        return self.session_manager.get_statistics()
