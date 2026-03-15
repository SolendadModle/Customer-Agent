"""
情感分析模块
基于关键词规则的轻量化情感分析，无需 GPU，适合生产环境快速部署。
当 sentiment_enabled=False 时，模块静默不执行任何分析。
"""

from typing import Dict, List, Tuple

from utils.logger import get_logger


# 负面情感关键词（中文电商场景）
NEGATIVE_KEYWORDS = [
    "差", "坏", "烂", "骗", "假", "劣质", "退款", "投诉", "举报", "不满",
    "失望", "愤怒", "气愤", "太差", "问题", "损坏", "破损", "缺货", "不发货",
    "欺骗", "虚假", "售后差", "服务差", "不退", "拒绝", "恶意",
]

# 正面情感关键词
POSITIVE_KEYWORDS = [
    "好", "棒", "赞", "满意", "感谢", "谢谢", "喜欢", "不错", "完美",
    "超级好", "非常好", "很好", "优秀", "给力", "推荐", "好评", "五星",
]

# 紧急关键词（触发转人工）
URGENCY_KEYWORDS = [
    "投诉", "举报", "骗子", "差评", "维权", "律师", "曝光", "媒体",
    "urgent", "immediately", "asap",
]


class SentimentAnalyzer:
    """基于关键词规则的情感分析器"""

    def __init__(self, enabled: bool = True,
                 negative_threshold: float = 0.6,
                 escalation_threshold: float = 0.8):
        self.logger = get_logger("SentimentAnalyzer")
        self.enabled = enabled
        self.negative_threshold = negative_threshold
        self.escalation_threshold = escalation_threshold

    def analyze(self, text: str) -> Dict:
        """
        分析文本情感

        Returns:
            {sentiment, score, intensity, enabled}
        """
        if not self.enabled or not text:
            return {"sentiment": "neutral", "score": 0.5, "intensity": "low", "enabled": False}

        text_lower = text.lower()

        neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        total = neg_count + pos_count or 1

        if neg_count > pos_count:
            sentiment = "negative"
            score = min(neg_count / total, 1.0)
        elif pos_count > neg_count:
            sentiment = "positive"
            score = min(pos_count / total, 1.0)
        else:
            sentiment = "neutral"
            score = 0.5

        intensity = "high" if score >= self.negative_threshold else "low"

        result = {
            "sentiment": sentiment,
            "score": score,
            "intensity": intensity,
            "enabled": True,
        }
        self.logger.debug(f"情感分析: {sentiment} ({score:.3f})")
        return result

    def should_escalate(self, text: str) -> Tuple[bool, str]:
        """
        判断是否需要转人工

        Returns:
            (should_escalate, reason)
        """
        if not self.enabled:
            return False, ""

        text_lower = text.lower()

        # 检查紧急关键词
        for kw in URGENCY_KEYWORDS:
            if kw in text_lower:
                return True, f"检测到紧急关键词：'{kw}'"

        # 检查高负面情感
        result = self.analyze(text)
        if result["sentiment"] == "negative" and result["score"] >= self.escalation_threshold:
            return True, f"检测到强烈负面情绪（分数: {result['score']:.2f}）"

        return False, ""

    def get_emotion_summary(self, conversation_history: List[Dict]) -> Dict:
        """
        汇总对话历史中的情感趋势

        Returns:
            {overall_sentiment, sentiment_trend, positive_count, negative_count, neutral_count}
        """
        if not self.enabled or not conversation_history:
            return {
                "overall_sentiment": "neutral",
                "sentiment_trend": "stable",
                "positive_count": 0, "negative_count": 0, "neutral_count": 0,
            }

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        timeline = []

        for msg in conversation_history:
            if msg.get("role") == "user":
                result = self.analyze(msg.get("content", ""))
                s = result["sentiment"]
                counts[s] += 1
                timeline.append(s)

        overall = max(counts, key=counts.get)

        trend = "stable"
        if len(timeline) >= 3:
            recent = timeline[-3:]
            if recent.count("negative") >= 2:
                trend = "declining"
            elif recent.count("positive") >= 2:
                trend = "improving"

        return {
            "overall_sentiment": overall,
            "sentiment_trend": trend,
            **{f"{k}_count": v for k, v in counts.items()},
        }
