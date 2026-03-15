"""
向量嵌入模块
使用 sentence-transformers 将文档转换为向量表示
当 sentence-transformers 未安装时，回退到基于字符哈希的伪嵌入（仅用于开发/测试）
"""

import os
import pickle
from typing import List, Optional

import numpy as np

from utils.logger import get_logger
from Agent.QianwenAgent.knowledge_base.kb_manager import Document


_logger = get_logger("DocumentEmbedder")

# 默认嵌入维度（sentence-transformers all-MiniLM-L6-v2）
DEFAULT_DIM = 384
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _hash_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """回退：基于哈希的伪向量（仅供开发测试）"""
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    # 重复/截断到目标维度
    repeats = dim // len(vec) + 1
    vec = np.tile(vec, repeats)[:dim]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class DocumentEmbedder:
    """文档向量嵌入器"""

    def __init__(self, model_name: str = DEFAULT_MODEL, dimension: int = DEFAULT_DIM,
                 batch_size: int = 32):
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.model = None
        self._use_fallback = False

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            _logger.info(f"嵌入模型加载成功: {model_name}，维度: {self.dimension}")
        except ImportError:
            _logger.warning("sentence-transformers 未安装，使用哈希伪嵌入（仅供测试）")
            self._use_fallback = True
        except Exception as e:
            _logger.warning(f"嵌入模型加载失败: {e}，使用哈希伪嵌入")
            self._use_fallback = True

    def embed_text(self, text: str) -> np.ndarray:
        if self._use_fallback or self.model is None:
            return _hash_embed(text, self.dimension)
        try:
            return self.model.encode(text, convert_to_numpy=True)
        except Exception as e:
            _logger.error(f"嵌入文本失败: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def embed_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        if self._use_fallback or self.model is None:
            return np.array([_hash_embed(t, self.dimension) for t in texts], dtype=np.float32)
        try:
            return self.model.encode(
                texts, batch_size=self.batch_size,
                show_progress_bar=show_progress, convert_to_numpy=True,
            )
        except Exception as e:
            _logger.error(f"批量嵌入失败: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)

    def embed_document(self, document: Document) -> np.ndarray:
        return self.embed_text(document.content)

    def embed_documents(self, documents: List[Document], show_progress: bool = False) -> np.ndarray:
        return self.embed_texts([doc.content for doc in documents], show_progress=show_progress)

    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                pickle.dump(embeddings, f)
            _logger.info(f"向量已保存: {file_path}")
        except Exception as e:
            _logger.error(f"保存向量失败: {e}")

    def load_embeddings(self, file_path: str) -> Optional[np.ndarray]:
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            _logger.error(f"加载向量失败: {e}")
            return None

    def get_embedding_dimension(self) -> int:
        return self.dimension
