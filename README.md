# 🤖 拼多多智能客服系统（融合增强版）

<div align="center">
  <img src="docs/设置.png" alt="系统配置界面" width="600">
  <p><em>拼多多智能客服系统 - 提升客服效率的智能化解决方案</em></p>
</div>

## 📖 项目简介

拼多多智能客服系统是一个专为电商平台设计的综合性客户服务管理工具。本系统通过AI技术和自动化流程，显著提高客服工作效率，实现智能回复的同时保留人工介入的灵活性，为商家提供完整的客服解决方案。

**v2.0 融合增强版**整合了三个项目的核心能力：
- **Customer-Agent**：拼多多电商客服框架 + PyQt6 GUI 桌面应用
- **RAG-based4**：千问大模型（QianwenAI / DashScope）+ 检索增强生成 + NLU + 情感分析 + 多轮对话 + 智能推荐
- **RAPTOR**：树形递归检索技术（Recursive Abstractive Processing for Tree-Organized Retrieval）

## ✨ 主要功能

### 🔐 账号管理
- 商家账号管理（支持多账号）
- 自动登录获取cookies
- 账号状态实时监控

### 💬 智能消息处理
- 实时消息监控与自动回复
- **双后端支持**：Coze API（原有）或千问大模型（新增）
- 支持自定义回复模板和关键词识别

### 🧠 千问大模型能力（新增）
- **NLU 意图识别**：自动识别用户意图（商品咨询、订单查询、退款等）
- **实体抽取**：提取订单号、金额、日期等关键实体
- **情感分析**：检测用户情绪，自动触发转人工服务
- **多轮对话管理**：维护会话历史和状态机
- **RAG 检索增强生成**：结合知识库内容生成精准回复
- **RAPTOR 树形检索**：多层次递归聚类知识检索，提升召回质量
- **智能推荐**：根据意图推荐相关商品/服务/解决方案

### 🔄 智能转接系统
- 基于关键词智能识别客户需求
- 情感分析自动检测负面情绪触发转人工
- 无缝衔接确保服务质量

### 📊 系统监控
- 实时日志记录
- 系统运行状态监控
- 详细的操作记录和统计

## 🚀 快速开始

### 环境要求
- Python 3.11+
- Windows 10/11 (推荐)
- 网络连接稳定

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/SolendadModle/Customer-Agent.git
   cd Customer-Agent
   ```

2. **安装依赖**
   ```bash
   pip install uv
   uv venv
   uv sync
   ```

3. **安装浏览器驱动**
   ```bash
   uv run playwright install chrome
   ```

4. **配置环境变量**（使用千问模式时）
   ```bash
   cp .env.example .env
   # 编辑 .env，填入 DASHSCOPE_API_KEY
   ```

## 📱 使用指南

### 启动系统
```bash
python app.py
```

### 配置流程

#### 方案一：Coze 模式（原有）
1. 在设置界面选择 Bot 类型为 `coze`
2. 填写 Coze API Base URL、Token 和 Bot ID

#### 方案二：千问大模型模式（推荐）
1. 在设置界面选择 Bot 类型为 `qianwen`
2. 填写阿里云 DashScope API Key（[获取地址](https://dashscope.aliyuncs.com/)）
3. 选择模型（qwen-turbo / qwen-plus / qwen-max）
4. 配置知识库路径（可选），将 FAQ 文件放入 `data/knowledge_base/faq.json`
5. 开启/关闭 RAPTOR 检索、情感分析、智能推荐等功能

### 知识库配置
在 `data/knowledge_base/faq.json` 中配置 FAQ 知识库：
```json
[
  {
    "question": "如何申请退款？",
    "answer": "您可以在订单页面点击申请退款，填写退款原因后提交即可。",
    "category": "退款"
  }
]
```

## 🛠️ 技术架构

- **前端界面**: PyQt6 + qfluentwidgets
- **后端逻辑**: Python 3.11+
- **AI 后端（Coze）**: Coze API
- **AI 后端（千问）**: 阿里云 DashScope / 千问大模型
- **RAG 检索**: FAISS 向量检索 + RAPTOR 树形递归检索
- **NLU**: 基于规则的意图识别 + 实体抽取
- **数据存储**: SQLite + JSON
- **浏览器自动化**: Playwright

## 📁 项目结构

```
Customer-Agent/
├── Agent/                      # AI 智能代理模块
│   ├── bot_factory.py          # 机器人工厂（支持 coze/qianwen）
│   ├── bot.py                  # 机器人抽象基类
│   ├── CozeAgent/              # Coze AI 代理（原有）
│   │   ├── bot.py
│   │   ├── conversation_manager.py
│   │   └── user_session.py
│   └── QianwenAgent/           # 千问 AI 代理（新增）
│       ├── bot.py              # QianwenBot 主入口
│       ├── raptor_retrieval.py # RAPTOR 树形递归检索
│       ├── nlp_utils.py        # NLP 工具函数
│       ├── preprocessing.py    # 文本预处理
│       ├── cache.py            # 内存缓存
│       ├── generation/         # 千问客户端 + RAG 生成器
│       │   ├── qianwen_client.py
│       │   └── rag_generator.py
│       ├── nlu/                # 意图识别 + 实体抽取
│       │   ├── intent_recognition.py
│       │   └── entity_extraction.py
│       ├── sentiment/          # 情感分析
│       │   └── sentiment_analyzer.py
│       ├── dialog/             # 多轮对话管理
│       │   ├── session_manager.py
│       │   └── dialog_handler.py
│       ├── recommendation/     # 智能推荐引擎
│       │   └── recommendation_engine.py
│       └── knowledge_base/     # 知识库管理
│           ├── kb_manager.py
│           ├── embeddings.py
│           └── retrieval.py
├── Channel/                    # 渠道接口模块
│   └── pinduoduo/
├── Message/                    # 消息处理模块
├── bridge/                     # 桥接模块
├── database/                   # 数据库模块
├── ui/                         # 用户界面
│   └── setting_ui.py           # 设置界面（含千问配置）
├── utils/                      # 工具函数
├── config/
│   └── config.yaml             # YAML 配置文件
├── data/                       # 运行时数据目录
│   ├── knowledge_base/         # 知识库文档
│   └── vector_store/           # 向量索引缓存
├── .env.example                # 环境变量示例
├── app.py                      # 应用程序入口
├── config.py                   # 配置管理
└── pyproject.toml              # 项目配置与依赖
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！如果您想参与项目开发：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，请给我们一个星标！</p>
</div>
