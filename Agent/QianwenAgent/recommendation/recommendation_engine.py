"""
智能推荐引擎
基于意图和对话历史，向用户推荐相关商品、服务或解决方案
"""

from typing import Any, Dict, List, Optional

from utils.logger import get_logger


# 默认商品/服务推荐目录（可在运行时扩展）
DEFAULT_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "prod_001",
        "name": "延长保障服务",
        "category": "services",
        "description": "购买延长保障，享受额外一年售后保修，轻松解决产品质量问题。",
        "tags": ["售后", "保修", "质量保障"],
    },
    {
        "id": "prod_002",
        "name": "优质会员计划",
        "category": "services",
        "description": "加入会员享受专属折扣、优先发货和专属客服通道。",
        "tags": ["会员", "折扣", "优先"],
    },
    {
        "id": "sol_001",
        "name": "极速退款通道",
        "category": "solutions",
        "description": "符合条件的订单可申请极速退款，24小时内原路退回。",
        "tags": ["退款", "快速", "退货"],
    },
    {
        "id": "sol_002",
        "name": "换货服务",
        "category": "solutions",
        "description": "商品质量问题可申请换货，无需退款，快速解决。",
        "tags": ["换货", "质量", "售后"],
    },
    {
        "id": "res_001",
        "name": "使用指南",
        "category": "resources",
        "description": "查看商品使用指南，了解产品功能和注意事项。",
        "tags": ["说明书", "指南", "使用方法"],
    },
    {
        "id": "res_002",
        "name": "常见问题解答",
        "category": "resources",
        "description": "浏览常见问题解答，快速解决常见疑问。",
        "tags": ["FAQ", "常见问题", "帮助"],
    },
]

# 意图 → 推荐类别映射
INTENT_CATEGORY_MAP = {
    "product_inquiry": "resources",
    "return_request": "solutions",
    "complaint": "solutions",
    "order_status": "services",
    "shipping_inquiry": "services",
    "payment_issue": "services",
    "general_question": "resources",
}

# 意图 → 推荐关键词映射（用于关键词匹配）
INTENT_KEYWORDS_MAP = {
    "product_inquiry": ["说明书", "使用方法", "产品功能"],
    "return_request": ["退款", "换货", "售后"],
    "complaint": ["解决", "赔偿", "处理"],
    "order_status": ["会员", "优先发货"],
    "shipping_inquiry": ["快递", "物流", "配送"],
    "payment_issue": ["优惠", "折扣", "付款"],
}


class RecommendationEngine:
    """基于意图的智能推荐引擎"""

    def __init__(self, enabled: bool = True,
                 max_recommendations: int = 3,
                 similarity_threshold: float = 0.0):
        self.logger = get_logger("RecommendationEngine")
        self.enabled = enabled
        self.max_recommendations = max_recommendations
        self.similarity_threshold = similarity_threshold
        self.catalog = list(DEFAULT_CATALOG)

    def recommend(self, query: str = "",
                  category: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        根据查询和类别返回推荐列表

        Args:
            query: 用户查询文本
            category: 目标类别过滤
            context: 额外上下文

        Returns:
            推荐条目列表
        """
        if not self.enabled or not self.catalog:
            return []

        filtered = self.catalog
        if category:
            filtered = [item for item in self.catalog if item["category"] == category]

        if not filtered:
            return []

        # 简单关键词打分
        scored: List[tuple] = []
        query_lower = query.lower()
        for item in filtered:
            score = sum(1 for tag in item.get("tags", []) if tag in query_lower)
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:self.max_recommendations]]
        self.logger.info(f"推荐 {len(results)} 条结果（query={query[:30]}）")
        return results

    def recommend_by_intent(self, intent: Optional[str],
                            entities: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """根据意图推荐"""
        if not intent or not self.enabled:
            return []

        category = INTENT_CATEGORY_MAP.get(intent)
        keywords = INTENT_KEYWORDS_MAP.get(intent, [])
        query = " ".join(keywords)
        return self.recommend(query=query, category=category)

    def recommend_based_on_history(self,
                                   conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据对话历史推荐"""
        if not conversation_history or not self.enabled:
            return []

        user_msgs = [
            msg["content"]
            for msg in conversation_history[-5:]
            if msg.get("role") == "user"
        ]
        combined_query = " ".join(user_msgs)
        return self.recommend(query=combined_query)

    def add_catalog_item(self, item: Dict[str, Any]):
        """动态添加推荐条目"""
        self.catalog.append(item)
        self.logger.info(f"添加推荐条目: {item.get('id', '?')}")
