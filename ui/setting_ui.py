# 设置界面

import json
import os
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QVBoxLayout, QWidget, QLabel, 
                            QFormLayout, QGroupBox, QMessageBox)
from PyQt6.QtGui import QFont
from qfluentwidgets import (CardWidget, SubtitleLabel, CaptionLabel, BodyLabel, 
                           PrimaryPushButton, PushButton, StrongBodyLabel, 
                           LineEdit, ComboBox, ScrollArea, FluentIcon as FIF,
                           InfoBar, InfoBarPosition, TextEdit, PasswordLineEdit,
                           TimePicker, SwitchButton)
from PyQt6.QtCore import QTime
from utils.logger import get_logger
from config import config


class CozeConfigCard(CardWidget):
    """Coze配置卡片"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        # 卡片标题
        title_label = StrongBodyLabel("Coze AI 配置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # 表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # API Base URL
        self.api_base_edit = LineEdit()
        self.api_base_edit.setPlaceholderText("https://api.coze.cn")
        self.api_base_edit.setText("https://api.coze.cn")
        form_layout.addRow("API Base URL:", self.api_base_edit)
        
        # API Token
        self.api_token_edit = PasswordLineEdit()
        self.api_token_edit.setPlaceholderText("输入您的 Coze API Token")
        form_layout.addRow("API Token:", self.api_token_edit)
        
        # Bot ID
        self.bot_id_edit = LineEdit()
        self.bot_id_edit.setPlaceholderText("输入您的 Bot ID")
        form_layout.addRow("Bot ID:", self.bot_id_edit)
                
        layout.addLayout(form_layout)
        
        # 说明文本
        description_label = CaptionLabel(
            "请在 Coze 平台获取您的 API Token 和 Bot ID。\n"
            "API Token 用于身份验证，Bot ID 用于指定使用的特定机器人。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)
    
    def getConfig(self) -> dict:
        """获取配置"""
        return {
            "coze_api_base": self.api_base_edit.text().strip() or "https://api.coze.cn",
            "coze_token": self.api_token_edit.text().strip(),
            "coze_bot_id": self.bot_id_edit.text().strip()
        }
    
    def setConfig(self, config: dict):
        """设置配置"""
        self.api_base_edit.setText(config.get("coze_api_base", "https://api.coze.cn"))
        self.api_token_edit.setText(config.get("coze_token", ""))
        self.bot_id_edit.setText(config.get("coze_bot_id", ""))


class QianwenConfigCard(CardWidget):
    """千问大模型配置卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("千问大模型配置 (QianwenAgent)")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)

        # DashScope API Key
        self.api_key_edit = PasswordLineEdit()
        self.api_key_edit.setPlaceholderText("输入 DashScope API Key")
        form_layout.addRow("DashScope API Key:", self.api_key_edit)

        # 模型选择
        self.model_combo = ComboBox()
        self.model_combo.addItems(["qwen-turbo", "qwen-plus", "qwen-max"])
        form_layout.addRow("模型:", self.model_combo)

        # 知识库路径
        self.kb_path_edit = LineEdit()
        self.kb_path_edit.setPlaceholderText("./data/knowledge_base")
        form_layout.addRow("知识库路径:", self.kb_path_edit)

        # 向量存储路径
        self.vector_path_edit = LineEdit()
        self.vector_path_edit.setPlaceholderText("./data/vector_store")
        form_layout.addRow("向量存储路径:", self.vector_path_edit)

        layout.addLayout(form_layout)

        # 功能开关区域
        switches_layout = QFormLayout()
        switches_layout.setSpacing(10)
        switches_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.raptor_switch = SwitchButton()
        self.raptor_switch.setChecked(True)
        switches_layout.addRow("启用 RAPTOR 树形检索:", self.raptor_switch)

        self.sentiment_switch = SwitchButton()
        self.sentiment_switch.setChecked(True)
        switches_layout.addRow("启用情感分析:", self.sentiment_switch)

        self.recommendation_switch = SwitchButton()
        self.recommendation_switch.setChecked(True)
        switches_layout.addRow("启用智能推荐:", self.recommendation_switch)

        self.nlu_switch = SwitchButton()
        self.nlu_switch.setChecked(True)
        switches_layout.addRow("启用 NLU（意图识别）:", self.nlu_switch)

        layout.addLayout(switches_layout)

        description_label = CaptionLabel(
            "使用阿里云千问大模型（DashScope）作为 AI 后端，整合 RAG 检索增强生成、\n"
            "RAPTOR 树形递归检索、情感分析、多轮对话和智能推荐能力。\n"
            "API Key 获取：https://dashscope.aliyuncs.com/"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {
            "dashscope_api_key": self.api_key_edit.text().strip(),
            "qianwen_model": self.model_combo.currentText(),
            "knowledge_base_path": self.kb_path_edit.text().strip() or "./data/knowledge_base",
            "vector_store_path": self.vector_path_edit.text().strip() or "./data/vector_store",
            "raptor_enabled": self.raptor_switch.isChecked(),
            "sentiment_enabled": self.sentiment_switch.isChecked(),
            "recommendation_enabled": self.recommendation_switch.isChecked(),
            "nlu_enabled": self.nlu_switch.isChecked(),
        }

    def setConfig(self, cfg: dict):
        self.api_key_edit.setText(cfg.get("dashscope_api_key", ""))
        model = cfg.get("qianwen_model", "qwen-turbo")
        idx = self.model_combo.findText(model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.kb_path_edit.setText(cfg.get("knowledge_base_path", "./data/knowledge_base"))
        self.vector_path_edit.setText(cfg.get("vector_store_path", "./data/vector_store"))
        self.raptor_switch.setChecked(cfg.get("raptor_enabled", True))
        self.sentiment_switch.setChecked(cfg.get("sentiment_enabled", True))
        self.recommendation_switch.setChecked(cfg.get("recommendation_enabled", True))
        self.nlu_switch.setChecked(cfg.get("nlu_enabled", True))


class BotTypeCard(CardWidget):
    """Bot 类型选择卡片"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        title_label = StrongBodyLabel("AI 后端选择")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)

        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.bot_type_combo = ComboBox()
        self.bot_type_combo.addItems(["coze", "qianwen"])
        form_layout.addRow("Bot 类型:", self.bot_type_combo)

        layout.addLayout(form_layout)

        description_label = CaptionLabel(
            "coze：使用 Coze 平台 API（原有方案）\n"
            "qianwen：使用阿里云千问大模型（RAG + RAPTOR，推荐）"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)

    def getConfig(self) -> dict:
        return {"bot_type": self.bot_type_combo.currentText()}

    def setConfig(self, cfg: dict):
        bt = cfg.get("bot_type", "coze")
        idx = self.bot_type_combo.findText(bt)
        if idx >= 0:
            self.bot_type_combo.setCurrentIndex(idx)


class BusinessHoursCard(CardWidget):
    """业务时间配置卡片"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
    
    def setupUI(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)
        
        # 卡片标题
        title_label = StrongBodyLabel("业务时间设置")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        layout.addWidget(title_label)
        
        # 表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(12)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # 开始时间
        self.start_time_picker = TimePicker()
        self.start_time_picker.setTime(QTime(8, 0))  # 默认8:00
        form_layout.addRow("开始时间:", self.start_time_picker)
        
        # 结束时间
        self.end_time_picker = TimePicker()
        self.end_time_picker.setTime(QTime(23, 0))  # 默认23:00
        form_layout.addRow("结束时间:", self.end_time_picker)
        
        layout.addLayout(form_layout)
        
        # 说明文本
        description_label = CaptionLabel(
            "设置AI客服的工作时间。在工作时间内，系统将自动响应客户消息。\n"
            "在非工作时间，系统将不会自动回复。"
        )
        description_label.setStyleSheet("color: #666; padding: 8px 0;")
        layout.addWidget(description_label)
    
    def getConfig(self) -> dict:
        """获取配置"""
        return {
            "businessHours": {
                "start": self.start_time_picker.getTime().toString("HH:mm"),
                "end": self.end_time_picker.getTime().toString("HH:mm")
            }
        }
    
    def setConfig(self, config: dict):
        """设置配置"""
        business_hours = config.get("businessHours", {})
        
        # 解析开始时间
        start_time_str = business_hours.get("start", "08:00")
        start_time = QTime.fromString(start_time_str, "HH:mm")
        if start_time.isValid():
            self.start_time_picker.setTime(start_time)
        
        # 解析结束时间
        end_time_str = business_hours.get("end", "23:00")
        end_time = QTime.fromString(end_time_str, "HH:mm")
        if end_time.isValid():
            self.end_time_picker.setTime(end_time)


class SettingUI(QFrame):
    """设置界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.logger = get_logger("SettingUI")
        self.setupUI()
        self.loadConfig()
        
        # 设置对象名
        self.setObjectName("设置")
    
    def setupUI(self):
        """设置主界面UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # 创建头部区域
        header_widget = self.createHeaderWidget()
        
        # 创建内容区域
        content_widget = self.createContentWidget()
        
        # 连接按钮信号
        self.save_btn.clicked.connect(self.onSaveConfig)
        self.reset_btn.clicked.connect(self.onResetConfig)
        
        # 添加到主布局
        main_layout.addWidget(header_widget)
        main_layout.addWidget(content_widget, 1)
    
    def createHeaderWidget(self):
        """创建头部区域"""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(20)
        
        # 标题
        title_label = SubtitleLabel("系统设置")
        title_label.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        
        # 描述
        description_label = CaptionLabel("配置AI客服的基本参数和工作时间")
        description_label.setStyleSheet("color: #666;")
        
        # 左侧标题区域
        title_area = QWidget()
        title_layout = QVBoxLayout(title_area)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addWidget(title_label)
        title_layout.addWidget(description_label)
        
        # 按钮区域
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)
        
        # 重置按钮
        self.reset_btn = PushButton("重置")
        self.reset_btn.setIcon(FIF.UPDATE)
        self.reset_btn.setFixedSize(80, 40)
        
        # 保存按钮
        self.save_btn = PrimaryPushButton("保存")
        self.save_btn.setIcon(FIF.SAVE)
        self.save_btn.setFixedSize(100, 40)
        
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.save_btn)
        
        # 添加到头部布局
        header_layout.addWidget(title_area)
        header_layout.addStretch()
        header_layout.addWidget(buttons_widget)
        
        return header_widget
    
    def createContentWidget(self):
        """创建内容区域"""
        # 滚动区域
        scroll_area = ScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 去除边框
        scroll_area.setStyleSheet("""
            ScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # 内容容器
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 创建配置卡片
        self.bot_type_card = BotTypeCard()
        self.coze_config_card = CozeConfigCard()
        self.qianwen_config_card = QianwenConfigCard()
        self.business_hours_card = BusinessHoursCard()
        
        # 添加到布局
        content_layout.addWidget(self.bot_type_card)
        content_layout.addWidget(self.coze_config_card)
        content_layout.addWidget(self.qianwen_config_card)
        content_layout.addWidget(self.business_hours_card)
        content_layout.addStretch()
        
        # 设置容器样式
        content_container.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
            }
        """)
        
        scroll_area.setWidget(content_container)
        
        return scroll_area
    
    def loadConfig(self):
        """从config模块加载配置"""
        try:
            loaded_config = {
                "bot_type": config.get("bot_type", "coze"),
                "coze_api_base": config.get("coze_api_base", "https://api.coze.cn"),
                "coze_token": config.get("coze_token", ""),
                "coze_bot_id": config.get("coze_bot_id", ""),
                "businessHours": config.get("businessHours", {"start": "08:00", "end": "23:00"}),
                "dashscope_api_key": config.get("dashscope_api_key", ""),
                "qianwen_model": config.get("qianwen_model", "qwen-turbo"),
                "knowledge_base_path": config.get("knowledge_base_path", "./data/knowledge_base"),
                "vector_store_path": config.get("vector_store_path", "./data/vector_store"),
                "raptor_enabled": config.get("raptor_enabled", True),
                "sentiment_enabled": config.get("sentiment_enabled", True),
                "recommendation_enabled": config.get("recommendation_enabled", True),
                "nlu_enabled": config.get("nlu_enabled", True),
            }
            
            # 验证并设置配置
            self._validateAndSetConfig(loaded_config)
            self.logger.info("配置加载成功")
            
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
            QMessageBox.warning(self, "加载失败", f"加载配置失败：{str(e)}")
            self._loadDefaultConfig()
    
    def _loadDefaultConfig(self):
        """加载默认配置"""
        default_config = {
            "bot_type": "coze",
            "coze_api_base": "https://api.coze.cn",
            "coze_token": "",
            "coze_bot_id": "",
            "businessHours": {"start": "08:00", "end": "23:00"},
            "dashscope_api_key": "",
            "qianwen_model": "qwen-turbo",
            "knowledge_base_path": "./data/knowledge_base",
            "vector_store_path": "./data/vector_store",
            "raptor_enabled": True,
            "sentiment_enabled": True,
            "recommendation_enabled": True,
            "nlu_enabled": True,
        }
        
        self.bot_type_card.setConfig(default_config)
        self.coze_config_card.setConfig(default_config)
        self.qianwen_config_card.setConfig(default_config)
        self.business_hours_card.setConfig(default_config)
        self.logger.info("已加载默认配置")
    
    def _validateAndSetConfig(self, config_data):
        """验证并设置配置"""
        validated_config = {
            "bot_type": config_data.get("bot_type", "coze"),
            "coze_api_base": config_data.get("coze_api_base", "https://api.coze.cn"),
            "coze_token": config_data.get("coze_token", ""),
            "coze_bot_id": config_data.get("coze_bot_id", ""),
            "businessHours": config_data.get("businessHours", {"start": "08:00", "end": "23:00"}),
            "dashscope_api_key": config_data.get("dashscope_api_key", ""),
            "qianwen_model": config_data.get("qianwen_model", "qwen-turbo"),
            "knowledge_base_path": config_data.get("knowledge_base_path", "./data/knowledge_base"),
            "vector_store_path": config_data.get("vector_store_path", "./data/vector_store"),
            "raptor_enabled": config_data.get("raptor_enabled", True),
            "sentiment_enabled": config_data.get("sentiment_enabled", True),
            "recommendation_enabled": config_data.get("recommendation_enabled", True),
            "nlu_enabled": config_data.get("nlu_enabled", True),
        }
        
        # 验证businessHours格式
        business_hours = validated_config["businessHours"]
        if not isinstance(business_hours, dict):
            business_hours = {"start": "08:00", "end": "23:00"}
            validated_config["businessHours"] = business_hours
        if "start" not in business_hours:
            business_hours["start"] = "08:00"
        if "end" not in business_hours:
            business_hours["end"] = "23:00"
        
        # 设置到界面
        self.bot_type_card.setConfig(validated_config)
        self.coze_config_card.setConfig(validated_config)
        self.qianwen_config_card.setConfig(validated_config)
        self.business_hours_card.setConfig(validated_config)
    
    def onSaveConfig(self):
        """保存配置到config模块"""
        try:
            # 获取配置
            bot_type_config = self.bot_type_card.getConfig()
            coze_config = self.coze_config_card.getConfig()
            qianwen_config = self.qianwen_config_card.getConfig()
            business_config = self.business_hours_card.getConfig()
            
            # 合并配置
            new_config = {**bot_type_config, **coze_config, **qianwen_config, **business_config}
            
            bot_type = new_config.get("bot_type", "coze")

            # 验证对应后端的必填项
            if bot_type == "coze":
                if not new_config.get("coze_token"):
                    QMessageBox.warning(self, "配置错误", "请输入 Coze API Token！")
                    return
                if not new_config.get("coze_bot_id"):
                    QMessageBox.warning(self, "配置错误", "请输入 Bot ID！")
                    return
            elif bot_type == "qianwen":
                if not new_config.get("dashscope_api_key"):
                    QMessageBox.warning(self, "配置错误", "请输入 DashScope API Key！")
                    return
            
            # 验证时间设置
            start_time = self.business_hours_card.start_time_picker.getTime()
            end_time = self.business_hours_card.end_time_picker.getTime()
            
            if start_time >= end_time:
                QMessageBox.warning(self, "时间设置错误", "开始时间必须早于结束时间！")
                return
            
            # 使用config模块保存配置
            config.update(new_config, save=True)
            
            self.logger.info("配置保存成功")
            
            # 显示成功消息
            InfoBar.success(
                title="保存成功",
                content="配置已保存！",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            QMessageBox.critical(self, "保存失败", f"保存配置时发生错误：{str(e)}")
    
    def onResetConfig(self):
        """重置配置"""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置所有配置吗？\n这将重新加载配置文件中的原始设置。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # 使用config模块重新加载配置文件
                config.reload()
                self.loadConfig()
                self.logger.info("配置已重置")
                
                InfoBar.success(
                    title="重置成功",
                    content="配置已重置为配置文件中的设置！",
                    orient=Qt.Orientation.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self
                )
            except Exception as e:
                self.logger.error(f"重置配置失败: {e}")
                QMessageBox.critical(self, "重置失败", f"重置配置失败：{str(e)}")
    
 