"""
知识库管理模块
管理 FAQ 和文档，支持关键词搜索和按类别检索
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger


class Document:
    """知识库文档"""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None,
                 doc_id: Optional[str] = None):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()

    def _generate_id(self) -> str:
        return hashlib.md5(self.content.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "content": self.content, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(content=data["content"], metadata=data.get("metadata", {}),
                   doc_id=data.get("doc_id"))


class KnowledgeBase:
    """知识库：加载、管理和检索文档"""

    def __init__(self, kb_path: str = "./data/knowledge_base"):
        self.kb_path = kb_path
        os.makedirs(kb_path, exist_ok=True)
        self.documents: List[Document] = []
        self.document_index: Dict[str, Document] = {}
        self.logger = get_logger("KnowledgeBase")
        self.logger.info("知识库初始化完成")

    def load_documents(self):
        """加载知识库目录中的所有文档"""
        kb_path = Path(self.kb_path)

        faq_file = kb_path / "faq.json"
        if faq_file.exists():
            self._load_faq(str(faq_file))

        docs_dir = kb_path / "documents"
        if docs_dir.exists():
            self._load_documents_directory(str(docs_dir))

        self.logger.info(f"知识库加载完成，共 {len(self.documents)} 条文档")

    def _load_faq(self, faq_path: str):
        """加载 FAQ JSON 文件"""
        try:
            with open(faq_path, "r", encoding="utf-8") as f:
                faq_data = json.load(f)
            for item in faq_data:
                q = item.get("question", "")
                a = item.get("answer", "")
                content = f"Q: {q}\nA: {a}"
                doc = Document(content=content, metadata={
                    "type": "faq",
                    "category": item.get("category", "general"),
                    "question": q,
                    "answer": a,
                })
                self.add_document(doc)
            self.logger.info(f"加载 FAQ {len(faq_data)} 条")
        except Exception as e:
            self.logger.error(f"加载 FAQ 失败: {e}")

    def _load_documents_directory(self, dir_path: str):
        """加载文档目录"""
        try:
            for fp in Path(dir_path).rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix == ".txt":
                    self._load_text_file(str(fp))
                elif fp.suffix == ".json":
                    self._load_json_document(str(fp))
        except Exception as e:
            self.logger.error(f"加载文档目录失败: {e}")

    def _load_text_file(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(content=content, metadata={
                "type": "text",
                "source_file": file_path,
                "filename": os.path.basename(file_path),
            })
            self.add_document(doc)
        except Exception as e:
            self.logger.error(f"加载文本文件失败 {file_path}: {e}")

    def _load_json_document(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                content = item.get("content") or item.get("text") or str(item)
                doc = Document(content=content, metadata={
                    "type": "json",
                    "source_file": file_path,
                })
                self.add_document(doc)
        except Exception as e:
            self.logger.error(f"加载 JSON 文档失败 {file_path}: {e}")

    def add_document(self, document: Document):
        self.documents.append(document)
        self.document_index[document.doc_id] = document

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.document_index.get(doc_id)

    def get_all_documents(self) -> List[Document]:
        return self.documents

    def search_by_keyword(self, keyword: str, top_k: int = 5) -> List[Document]:
        kw_lower = keyword.lower()
        return [doc for doc in self.documents if kw_lower in doc.content.lower()][:top_k]

    def get_documents_by_category(self, category: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.get("category") == category]

    def get_statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"total_documents": len(self.documents), "document_types": {}}
        for doc in self.documents:
            dtype = doc.metadata.get("type", "unknown")
            stats["document_types"][dtype] = stats["document_types"].get(dtype, 0) + 1
        return stats
