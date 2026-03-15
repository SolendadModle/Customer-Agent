"""
会话管理模块
管理用户多轮对话的会话状态、历史消息和上下文信息
"""

import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.logger import get_logger


def _generate_session_id(user_id: Optional[str] = None) -> str:
    """生成唯一会话 ID"""
    base = user_id or ""
    return f"sess_{base}_{uuid.uuid4().hex[:8]}"


class ConversationSession:
    """单个对话会话"""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.state: str = "initial"
        self.history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """追加一条消息到历史"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        })
        self.last_activity = datetime.utcnow()

    def get_history(self, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取历史消息（可限制条数）"""
        return self.history[-max_turns:] if max_turns else self.history

    def set_state(self, state: str):
        """设置会话状态"""
        self.state = state
        self.last_activity = datetime.utcnow()

    def update_context(self, key: str, value: Any):
        """更新上下文变量"""
        self.context[key] = value
        self.last_activity = datetime.utcnow()

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def is_expired(self, timeout_seconds: int) -> bool:
        elapsed = (datetime.utcnow() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "state": self.state,
            "history": self.history,
            "context": self.context,
        }


class SessionManager:
    """多会话管理器"""

    def __init__(self, session_timeout: int = 1800, max_history_turns: int = 10):
        self.session_timeout = session_timeout
        self.max_history_turns = max_history_turns
        self.sessions: Dict[str, ConversationSession] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.logger = get_logger("SessionManager")

    def create_session(self, user_id: Optional[str] = None) -> ConversationSession:
        """创建新会话"""
        session_id = _generate_session_id(user_id)
        session = ConversationSession(session_id, user_id)
        self.sessions[session_id] = session
        if user_id:
            self.user_sessions[user_id].append(session_id)
        self.logger.info(f"创建会话: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话（已过期返回 None）"""
        session = self.sessions.get(session_id)
        if session:
            if session.is_expired(self.session_timeout):
                self.logger.info(f"会话已过期: {session_id}")
                self.delete_session(session_id)
                return None
            return session
        return None

    def get_or_create_session(self, session_id: Optional[str] = None,
                              user_id: Optional[str] = None) -> ConversationSession:
        """获取已有会话，不存在则创建新会话"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session(user_id)

    def delete_session(self, session_id: str):
        """删除会话"""
        session = self.sessions.pop(session_id, None)
        if session and session.user_id:
            try:
                self.user_sessions[session.user_id].remove(session_id)
            except ValueError:
                pass
        self.logger.info(f"删除会话: {session_id}")

    def cleanup_expired_sessions(self):
        """清理所有过期会话"""
        expired = [sid for sid, s in self.sessions.items()
                   if s.is_expired(self.session_timeout)]
        for sid in expired:
            self.delete_session(sid)
        if expired:
            self.logger.info(f"清理 {len(expired)} 个过期会话")

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_sessions": len(self.sessions),
            "active_users": len([uid for uid, sids in self.user_sessions.items() if sids]),
            "session_timeout": self.session_timeout,
        }
