"""
FAISS 向量检索服务
将知识库文档索引到 FAISS，支持余弦相似度检索
当 faiss-cpu 未安装时，回退到 numpy 暴力搜索
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from Agent.QianwenAgent.knowledge_base.kb_manager import Document, KnowledgeBase
from Agent.QianwenAgent.knowledge_base.embeddings import DocumentEmbedder


_logger = get_logger("RetrievalService")


def _cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """计算 query_vec 与 matrix 每行的余弦相似度"""
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(len(matrix))
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-9
    return (matrix @ query_vec) / (norms * q_norm)


class RetrievalService:
    """知识库向量检索服务"""

    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None,
                 vector_store_path: str = "./data/vector_store",
                 top_k: int = 5,
                 similarity_threshold: float = 0.3,
                 max_context_length: int = 2000):
        self.kb = knowledge_base or KnowledgeBase()
        self.embedder = DocumentEmbedder()
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length
        os.makedirs(vector_store_path, exist_ok=True)

        self._faiss_index = None
        self._np_embeddings: Optional[np.ndarray] = None
        self.document_ids: List[str] = []

        _logger.info("向量检索服务初始化完成")

    # ------------------------------------------------------------------ #
    #  Index management                                                     #
    # ------------------------------------------------------------------ #

    def build_index(self, force_rebuild: bool = False):
        """构建或加载索引"""
        index_file = os.path.join(self.vector_store_path, "faiss_index.bin")
        doc_ids_file = os.path.join(self.vector_store_path, "document_ids.pkl")
        emb_file = os.path.join(self.vector_store_path, "embeddings.npy")

        if not force_rebuild and os.path.exists(doc_ids_file) and (
            os.path.exists(index_file) or os.path.exists(emb_file)
        ):
            try:
                self._load_index(index_file, doc_ids_file, emb_file)
                _logger.info("已加载现有索引")
                return
            except Exception as e:
                _logger.warning(f"加载索引失败，重新构建: {e}")

        _logger.info("开始构建向量索引...")
        if not self.kb.documents:
            self.kb.load_documents()

        documents = self.kb.get_all_documents()
        if not documents:
            _logger.warning("知识库为空，无法构建索引")
            return

        embeddings = self.embedder.embed_documents(documents, show_progress=True)
        self.document_ids = [doc.doc_id for doc in documents]

        # 尝试 FAISS
        try:
            import faiss
            import faiss as _faiss
            emb_norm = embeddings.copy().astype(np.float32)
            _faiss.normalize_L2(emb_norm)
            dim = emb_norm.shape[1]
            self._faiss_index = _faiss.IndexFlatIP(dim)
            self._faiss_index.add(emb_norm)
            _faiss.write_index(self._faiss_index, index_file)
            _logger.info(f"FAISS 索引构建完成，共 {len(documents)} 条")
        except ImportError:
            _logger.warning("faiss-cpu 未安装，使用 numpy 暴力搜索")
            self._np_embeddings = embeddings.astype(np.float32)
            np.save(emb_file, self._np_embeddings)

        with open(doc_ids_file, "wb") as f:
            pickle.dump(self.document_ids, f)

    def _load_index(self, index_file: str, doc_ids_file: str, emb_file: str):
        with open(doc_ids_file, "rb") as f:
            self.document_ids = pickle.load(f)
        if os.path.exists(index_file):
            try:
                import faiss
                self._faiss_index = faiss.read_index(index_file)
                return
            except ImportError:
                pass
        if os.path.exists(emb_file):
            self._np_embeddings = np.load(emb_file)

    # ------------------------------------------------------------------ #
    #  Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """检索与 query 最相关的文档"""
        if self._faiss_index is None and self._np_embeddings is None:
            _logger.warning("索引未构建，尝试构建...")
            self.build_index()

        k = top_k or self.top_k
        query_emb = self.embedder.embed_text(query).astype(np.float32)

        if self._faiss_index is not None:
            return self._retrieve_faiss(query_emb, k)
        elif self._np_embeddings is not None:
            return self._retrieve_numpy(query_emb, k)
        return []

    def _retrieve_faiss(self, query_emb: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        import faiss
        q = query_emb.reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self._faiss_index.search(q, min(k, self._faiss_index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or score < self.similarity_threshold:
                continue
            doc = self.kb.get_document(self.document_ids[idx])
            if doc:
                results.append((doc, float(score)))
        return results

    def _retrieve_numpy(self, query_emb: np.ndarray, k: int) -> List[Tuple[Document, float]]:
        sims = _cosine_similarity_matrix(query_emb, self._np_embeddings)
        top_indices = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < self.similarity_threshold:
                continue
            doc = self.kb.get_document(self.document_ids[idx])
            if doc:
                results.append((doc, score))
        return results

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> str:
        """检索并格式化为 RAG 上下文字符串"""
        results = self.retrieve(query, top_k)
        if not results:
            return "暂无相关知识库内容。"

        parts = []
        total_len = 0
        for doc, score in results:
            content = doc.content
            if total_len + len(content) > self.max_context_length:
                remaining = self.max_context_length - total_len
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            parts.append(f"[相关度: {score:.3f}]\n{content}")
            total_len += len(content)

        return "\n\n---\n\n".join(parts)

    def get_index_stats(self) -> Dict[str, Any]:
        if self._faiss_index is not None:
            return {
                "status": "ready",
                "backend": "faiss",
                "total_documents": self._faiss_index.ntotal,
                "dimension": self._faiss_index.d,
            }
        if self._np_embeddings is not None:
            return {
                "status": "ready",
                "backend": "numpy",
                "total_documents": len(self._np_embeddings),
            }
        return {"status": "not_built"}
