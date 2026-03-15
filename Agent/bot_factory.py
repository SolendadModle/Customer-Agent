"""
Bot 工厂
根据 config.json 中的 bot_type 创建对应 Bot 实例
支持：coze（默认）、qianwen
"""
from config import config


def create_bot():
    """
    创建一个 bot 实例

    bot_type 配置项：
      - "coze"     使用 Coze API（默认）
      - "qianwen"  使用千问大模型 + RAG + RAPTOR

    :return: bot 实例
    """
    bot_type = config.get("bot_type", "coze")
    if bot_type == "coze":
        from Agent.CozeAgent.bot import CozeBot
        return CozeBot()
    elif bot_type == "qianwen":
        from Agent.QianwenAgent.bot import QianwenBot
        return QianwenBot()
    else:
        raise RuntimeError(f"不支持的 bot_type: {bot_type}，请在 config.json 中设置为 'coze' 或 'qianwen'")