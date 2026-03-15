"""
意图识别模块
基于关键词规则与 TF-IDF 相似度的轻量化意图识别，适配电商客服场景
"""

import re
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger


# 电商客服意图词典
INTENT_PATTERNS: Dict[str, List[str]] = {
    "greeting": [
        "你好", "您好", "hi", "hello", "早上好", "下午好", "晚上好", "在吗", "有人吗",
    ],
    "farewell": [
        "再见", "拜拜", "goodbye", "bye", "谢谢", "感谢", "好的谢谢", "没问题了",
    ],
    "product_inquiry": [
        "商品", "产品", "这个怎么样", "有没有", "规格", "尺寸", "颜色", "款式",
        "材质", "成分", "功能", "参数", "介绍", "详情", "说明",
    ],
    "order_status": [
        "订单", "我的订单", "查询订单", "订单号", "发货了吗", "什么时候发货",
        "单号", "快递", "物流", "配送", "几天到", "到哪了",
    ],
    "return_request": [
        "退款", "退货", "换货", "不想要了", "质量问题", "损坏", "破损",
        "申请退款", "退换", "不合适", "尺码不对",
    ],
    "complaint": [
        "投诉", "举报", "差评", "骗子", "虚假", "不满意", "太差了",
        "欺诈", "假货", "维权",
    ],
    "shipping_inquiry": [
        "快递", "物流", "运费", "邮费", "包邮", "几天", "多久", "什么快递",
        "配送时间", "运输",
    ],
    "payment_issue": [
        "支付", "付款", "付不了", "支付失败", "优惠券", "红包", "折扣",
        "怎么付", "能用什么付",
    ],
    "feedback": [
        "建议", "意见", "反馈", "希望", "可以改进", "体验",
    ],
    "general_question": [
        "怎么", "如何", "什么", "为什么", "能不能", "可以吗", "帮我", "请问",
    ],
}


class IntentRecognizer:
    """基于规则 + TF-IDF 的电商客服意图识别"""

    def __init__(self, threshold: float = 0.6):
        self.logger = get_logger("IntentRecognizer")
        self.threshold = threshold
        self.intent_patterns = INTENT_PATTERNS
        self.logger.info("意图识别器初始化完成")

    def recognize(self, text: str) -> Tuple[str, float]:
        """
        识别文本意图

        Args:
            text: 用户输入文本

        Returns:
            (intent, confidence) 元组
        """
        if not text or not text.strip():
            return "general_question", 0.0

        text_lower = text.lower().strip()

        best_intent = "general_question"
        best_score = 0.0

        for intent, keywords in self.intent_patterns.items():
            score = self._match_score(text_lower, keywords)
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_score < self.threshold:
            best_intent = "general_question"

        self.logger.debug(f"意图识别: {best_intent}（置信度: {best_score:.3f}）")
        return best_intent, best_score

    def _match_score(self, text: str, keywords: List[str]) -> float:
        """计算文本与关键词列表的匹配分数"""
        if not text:
            return 0.0

        matched_keywords = [kw for kw in keywords if kw.lower() in text]
        if not matched_keywords:
            return 0.0

        # 使用匹配到的最长关键词占文本比例作为分数，并给一个基础加成
        best_kw_len = max(len(kw) for kw in matched_keywords)
        # 基础分 0.6 + 关键词长度/文本长度 * 0.4
        ratio = min(best_kw_len / max(len(text), 1), 1.0)
        score = 0.6 + ratio * 0.4
        return min(score, 1.0)

    def get_supported_intents(self) -> List[str]:
        """返回支持的意图列表"""
        return list(self.intent_patterns.keys())
