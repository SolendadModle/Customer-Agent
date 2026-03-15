"""
RAPTOR 树形递归检索模块
Recursive Abstractive Processing for Tree-Organized Retrieval

核心思路：
1. 对知识库文档进行递归聚类（使用 K-Means）
2. 对每个聚类生成摘要（使用千问大模型）
3. 构建多层树形知识索引（叶节点 → 中间节点 → 根节点）
4. 检索时支持从叶节点细节到根节点摘要的多层次召回
"""

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from Agent.QianwenAgent.knowledge_base.kb_manager import Document, KnowledgeBase
from Agent.QianwenAgent.knowledge_base.embeddings import DocumentEmbedder


_logger = get_logger("RAPTOR")


# --------------------------------------------------------------------------- #
#  数据结构                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class TreeNode:
    """RAPTOR 树形节点"""
    node_id: str
    content: str                          # 节点文本（叶节点=原文，中间节点=摘要）
    level: int                            # 层级（0=叶节点，1,2,...=聚类摘要）
    children: List["TreeNode"] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "level": self.level,
            "children": [c.node_id for c in self.children],
            "metadata": self.metadata,
        }


def _make_node_id(content: str, level: int) -> str:
    digest = hashlib.md5(f"{level}:{content}".encode("utf-8")).hexdigest()[:12]
    return f"node_l{level}_{digest}"


# --------------------------------------------------------------------------- #
#  KMeans 辅助（优先使用 sklearn，否则 numpy 实现）                            #
# --------------------------------------------------------------------------- #

def _kmeans_cluster(embeddings: np.ndarray, n_clusters: int,
                    max_iter: int = 100) -> np.ndarray:
    """返回每个样本的簇标签 (shape: [n,])"""
    if len(embeddings) <= n_clusters:
        return np.arange(len(embeddings))

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=10, max_iter=max_iter, random_state=42)
        return km.fit_predict(embeddings)
    except ImportError:
        _logger.warning("sklearn 未安装，使用简单 numpy K-Means")
        return _numpy_kmeans(embeddings, n_clusters, max_iter)


def _numpy_kmeans(embeddings: np.ndarray, n_clusters: int, max_iter: int) -> np.ndarray:
    """简单 numpy K-Means 实现"""
    n = len(embeddings)
    centers = embeddings[np.random.choice(n, n_clusters, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # 分配
        dists = np.linalg.norm(embeddings[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # 更新中心
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = embeddings[mask].mean(axis=0)
    return labels


# --------------------------------------------------------------------------- #
#  RaptorTree                                                                  #
# --------------------------------------------------------------------------- #

class RaptorTree:
    """RAPTOR 树形索引构建器"""

    def __init__(self, qianwen_client=None,
                 embedder: Optional[DocumentEmbedder] = None,
                 max_levels: int = 3,
                 cluster_size: int = 5,
                 summary_max_tokens: int = 200):
        """
        Args:
            qianwen_client: QianwenClient 实例，用于生成摘要
            embedder: 向量嵌入器
            max_levels: 最大树层数（不含叶节点层）
            cluster_size: 每个聚类目标大小
            summary_max_tokens: 摘要最大 token 数
        """
        self.qianwen_client = qianwen_client
        self.embedder = embedder or DocumentEmbedder()
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.summary_max_tokens = summary_max_tokens

        # 按层存储节点
        self.levels: Dict[int, List[TreeNode]] = {}
        self.all_nodes: Dict[str, TreeNode] = {}

    # ------------------------------------------------------------------ #
    #  构建                                                                #
    # ------------------------------------------------------------------ #

    def build(self, documents: List[Document]) -> List[TreeNode]:
        """
        从文档列表构建 RAPTOR 树

        Returns:
            根节点列表（最高层）
        """
        if not documents:
            _logger.warning("文档列表为空，无法构建 RAPTOR 树")
            return []

        _logger.info(f"开始构建 RAPTOR 树，共 {len(documents)} 条文档")

        # Level 0: 叶节点
        leaf_nodes = self._create_leaf_nodes(documents)
        self.levels[0] = leaf_nodes
        for node in leaf_nodes:
            self.all_nodes[node.node_id] = node

        current_nodes = leaf_nodes
        for level in range(1, self.max_levels + 1):
            if len(current_nodes) <= self.cluster_size:
                _logger.info(f"层 {level} 节点数 ({len(current_nodes)}) <= cluster_size，停止递归")
                break

            upper_nodes = self._build_level(current_nodes, level)
            self.levels[level] = upper_nodes
            for node in upper_nodes:
                self.all_nodes[node.node_id] = node
            current_nodes = upper_nodes
            _logger.info(f"层 {level} 构建完成，共 {len(upper_nodes)} 个节点")

        root_level = max(self.levels.keys())
        return self.levels[root_level]

    def _create_leaf_nodes(self, documents: List[Document]) -> List[TreeNode]:
        """从文档创建叶节点并生成嵌入"""
        _logger.info("生成叶节点嵌入...")
        embeddings = self.embedder.embed_documents(documents)
        nodes = []
        for doc, emb in zip(documents, embeddings):
            node = TreeNode(
                node_id=_make_node_id(doc.content, 0),
                content=doc.content,
                level=0,
                embedding=emb,
                metadata=doc.metadata,
            )
            nodes.append(node)
        return nodes

    def _build_level(self, nodes: List[TreeNode], level: int) -> List[TreeNode]:
        """对当前层节点聚类并生成摘要节点"""
        embeddings = np.array([n.embedding for n in nodes if n.embedding is not None],
                               dtype=np.float32)
        if len(embeddings) == 0:
            return []

        n_clusters = max(1, len(nodes) // self.cluster_size)
        labels = _kmeans_cluster(embeddings, n_clusters)

        upper_nodes = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_nodes = [nodes[i] for i in range(len(nodes)) if cluster_mask[i]]
            if not cluster_nodes:
                continue

            # 生成聚类摘要
            summary = self._summarize_cluster(cluster_nodes)
            summary_emb = self.embedder.embed_text(summary)

            parent = TreeNode(
                node_id=_make_node_id(summary, level),
                content=summary,
                level=level,
                children=cluster_nodes,
                embedding=summary_emb,
                metadata={"cluster_id": cluster_id, "child_count": len(cluster_nodes)},
            )
            upper_nodes.append(parent)

        return upper_nodes

    def _summarize_cluster(self, nodes: List[TreeNode]) -> str:
        """使用千问模型为聚类节点生成摘要"""
        combined = "\n\n".join(n.content[:500] for n in nodes[:10])
        prompt = (
            f"请对以下客服知识库内容进行简洁摘要（不超过 {self.summary_max_tokens} 字）：\n\n"
            f"{combined}\n\n摘要："
        )

        if self.qianwen_client:
            try:
                result = self.qianwen_client.generate(
                    prompt=prompt,
                    max_tokens=self.summary_max_tokens,
                )
                if result:
                    return result.strip()
            except Exception as e:
                _logger.error(f"千问摘要生成失败: {e}")

        # 回退：截取前 N 个节点内容拼接
        return " | ".join(n.content[:100] for n in nodes[:3])

    # ------------------------------------------------------------------ #
    #  持久化                                                              #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """保存树到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "levels": {
                        lv: [n.to_dict() for n in nodes]
                        for lv, nodes in self.levels.items()
                    },
                    "embeddings": {nid: n.embedding for nid, n in self.all_nodes.items()},
                    "contents": {nid: n.content for nid, n in self.all_nodes.items()},
                    "metadata": {nid: n.metadata for nid, n in self.all_nodes.items()},
                }, f)
            _logger.info(f"RAPTOR 树已保存: {path}")
        except Exception as e:
            _logger.error(f"保存 RAPTOR 树失败: {e}")

    def load(self, path: str) -> bool:
        """从文件加载树"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            emb_map = data.get("embeddings", {})
            content_map = data.get("contents", {})
            meta_map = data.get("metadata", {})

            self.levels = {}
            self.all_nodes = {}

            for lv_str, node_dicts in data["levels"].items():
                lv = int(lv_str)
                level_nodes = []
                for nd in node_dicts:
                    nid = nd["node_id"]
                    node = TreeNode(
                        node_id=nid,
                        content=content_map.get(nid, nd.get("content", "")),
                        level=lv,
                        embedding=emb_map.get(nid),
                        metadata=meta_map.get(nid, {}),
                    )
                    level_nodes.append(node)
                    self.all_nodes[nid] = node
                self.levels[lv] = level_nodes

            _logger.info(f"RAPTOR 树已加载: {path}")
            return True
        except Exception as e:
            _logger.error(f"加载 RAPTOR 树失败: {e}")
            return False


# --------------------------------------------------------------------------- #
#  RaptorRetriever                                                             #
# --------------------------------------------------------------------------- #

class RaptorRetriever:
    """
    RAPTOR 树形递归检索器
    - 叶节点层：细节级检索（精确内容）
    - 中间节点层：主题级检索（聚类摘要）
    - 根节点层：全局级检索（最高层摘要）
    """

    def __init__(self, raptor_tree: Optional[RaptorTree] = None,
                 embedder: Optional[DocumentEmbedder] = None,
                 leaf_top_k: int = 3,
                 node_top_k: int = 2,
                 similarity_threshold: float = 0.2,
                 max_context_length: int = 2000):
        self.raptor_tree = raptor_tree or RaptorTree()
        self.embedder = embedder or self.raptor_tree.embedder
        self.leaf_top_k = leaf_top_k
        self.node_top_k = node_top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length

    def retrieve(self, query: str) -> List[Tuple[TreeNode, float]]:
        """
        多层次树形检索

        Returns:
            [(TreeNode, score)] 按分数降序排列
        """
        if not self.raptor_tree.all_nodes:
            _logger.warning("RAPTOR 树为空，无法检索")
            return []

        query_emb = self.embedder.embed_text(query).astype(np.float32)
        all_results: List[Tuple[TreeNode, float]] = []

        for level, nodes in sorted(self.raptor_tree.levels.items()):
            top_k = self.leaf_top_k if level == 0 else self.node_top_k
            level_results = self._search_level(query_emb, nodes, top_k)
            all_results.extend(level_results)

        # 按分数降序，去重
        seen_ids = set()
        unique_results = []
        for node, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            if node.node_id not in seen_ids and score >= self.similarity_threshold:
                seen_ids.add(node.node_id)
                unique_results.append((node, score))

        return unique_results

    def _search_level(self, query_emb: np.ndarray, nodes: List[TreeNode],
                      top_k: int) -> List[Tuple[TreeNode, float]]:
        """在某一层的节点中搜索"""
        candidates = [n for n in nodes if n.embedding is not None]
        if not candidates:
            return []

        embeddings = np.array([n.embedding for n in candidates], dtype=np.float32)
        sims = self._cosine_similarities(query_emb, embeddings)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(candidates[i], float(sims[i])) for i in top_indices]

    @staticmethod
    def _cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return np.zeros(len(matrix))
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-9
        return (matrix @ query) / (norms * q_norm)

    def retrieve_context(self, query: str) -> str:
        """检索并格式化为 RAG 上下文字符串"""
        results = self.retrieve(query)
        if not results:
            return "暂无相关知识库内容（RAPTOR 检索）。"

        parts = []
        total_len = 0
        for node, score in results:
            level_label = "叶节点" if node.level == 0 else f"摘要(层{node.level})"
            content = f"[{level_label} 相关度: {score:.3f}]\n{node.content}"
            if total_len + len(content) > self.max_context_length:
                remaining = self.max_context_length - total_len
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break
            parts.append(content)
            total_len += len(content)

        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """返回树形统计信息"""
        return {
            "levels": {
                lv: len(nodes)
                for lv, nodes in self.raptor_tree.levels.items()
            },
            "total_nodes": len(self.raptor_tree.all_nodes),
        }


# --------------------------------------------------------------------------- #
#  便利工厂函数                                                                #
# --------------------------------------------------------------------------- #

def build_raptor_retriever(knowledge_base: KnowledgeBase,
                           qianwen_client=None,
                           vector_store_path: str = "./data/vector_store",
                           max_levels: int = 3,
                           cluster_size: int = 5,
                           force_rebuild: bool = False) -> RaptorRetriever:
    """
    从知识库构建并返回 RaptorRetriever

    Args:
        knowledge_base: 知识库实例
        qianwen_client: 千问客户端（用于生成摘要）
        vector_store_path: 索引存储路径
        max_levels: 最大树层数
        cluster_size: 聚类大小
        force_rebuild: 强制重建

    Returns:
        已构建的 RaptorRetriever
    """
    embedder = DocumentEmbedder()
    tree = RaptorTree(
        qianwen_client=qianwen_client,
        embedder=embedder,
        max_levels=max_levels,
        cluster_size=cluster_size,
    )

    tree_path = os.path.join(vector_store_path, "raptor_tree.pkl")

    if not force_rebuild and tree.load(tree_path):
        _logger.info("已加载现有 RAPTOR 树")
    else:
        if not knowledge_base.documents:
            knowledge_base.load_documents()
        documents = knowledge_base.get_all_documents()
        tree.build(documents)
        tree.save(tree_path)

    return RaptorRetriever(raptor_tree=tree, embedder=embedder)
