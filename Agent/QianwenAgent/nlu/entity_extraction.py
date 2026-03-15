"""
实体抽取模块
使用正则规则从电商客服对话中抽取关键实体（订单号、金额、日期、电话等）
"""

import re
from typing import Any, Dict, List, Tuple

from utils.logger import get_logger
from Agent.QianwenAgent.nlu.intent_recognition import IntentRecognizer


# 正则规则
ENTITY_PATTERNS: Dict[str, List[str]] = {
    "ORDER_ID": [
        r"\b[A-Z]{2,3}\d{6,12}\b",   # 字母+数字订单号（如 PDD12345678）
        r"\b\d{10,18}\b",              # 纯数字长订单号
        r"订单[号码]?\s*[:：]?\s*(\d{6,18})",
    ],
    "PHONE": [
        r"1[3-9]\d{9}",
        r"\d{3}[-.\s]\d{4}[-.\s]\d{4}",
    ],
    "MONEY": [
        r"[\￥¥]\d+(?:\.\d{2})?",
        r"\d+(?:\.\d{2})?\s*元",
    ],
    "DATE": [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}月\d{1,2}[日号]",
        r"(?:今天|明天|后天|昨天)",
    ],
    "PRODUCT_CODE": [
        r"\b[A-Z]{2,4}[-_]\d{3,8}\b",
        r"SKU\s*[:：]?\s*\w{4,12}",
    ],
}


class EntityExtractor:
    """基于正则规则的实体抽取器"""

    def __init__(self):
        self.logger = get_logger("EntityExtractor")
        self.patterns = ENTITY_PATTERNS

    def extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        从文本中抽取实体

        Args:
            text: 输入文本

        Returns:
            {实体类型: [{text, label, start, end}]}
        """
        if not text:
            return {}

        entities: Dict[str, List[Dict[str, Any]]] = {}

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    for m in re.finditer(pattern, text, re.IGNORECASE):
                        # 取第一个捕获组（如果存在），否则取全匹配
                        matched_text = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group()
                        info = {
                            "text": matched_text,
                            "label": entity_type,
                            "start": m.start(),
                            "end": m.end(),
                        }
                        entities.setdefault(entity_type, []).append(info)
                except Exception as e:
                    self.logger.error(f"实体抽取正则异常 [{entity_type}]: {e}")

        # 去重
        for etype in entities:
            seen = set()
            unique = []
            for ent in entities[etype]:
                key = ent["text"]
                if key not in seen:
                    seen.add(key)
                    unique.append(ent)
            entities[etype] = unique

        self.logger.debug(f"抽取实体: {entities}")
        return entities

    def get_supported_entities(self) -> List[str]:
        """返回支持的实体类型"""
        return list(self.patterns.keys())


class NLUService:
    """NLU 联合服务：意图识别 + 实体抽取"""

    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()
        self.logger = get_logger("NLUService")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        分析文本，返回意图和实体

        Returns:
            {text, intent: {name, confidence}, entities}
        """
        intent, confidence = self.intent_recognizer.recognize(text)
        entities = self.entity_extractor.extract(text)

        result = {
            "text": text,
            "intent": {"name": intent, "confidence": confidence},
            "entities": entities,
        }
        self.logger.info(f"NLU 分析 - 意图: {intent}({confidence:.3f}), 实体: {len(entities)} 类")
        return result
