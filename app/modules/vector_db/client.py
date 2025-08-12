import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from app.common.logger import logger
from app.common.exceptions import RetrievalException


class FaissClient:
    def __init__(self, index_path: str = "../data/faiss_index", dim: int = 1024):
        """
        Faiss向量库客户端（本地索引）
        :param index_path: 索引文件保存路径
        :param dim: 向量维度（BGE-large-zh模型输出为1024维）
        """
        self.index_path = Path(index_path)
        self.dim = dim
        self.index = None  # Faiss索引对象
        self.metadata = []  # 存储文档元数据（与向量一一对应）
        self._load_or_create_index()

    def _load_or_create_index(self):
        """加载已有索引或创建新索引（IVF_FLAT类型，适合百万级数据）"""
        # 检查索引文件和元数据文件
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.pkl"

        if index_file.exists() and meta_file.exists():
            # 加载已有索引
            self.index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"加载Faiss索引成功，共{len(self.metadata)}条数据")
        else:
            # 创建新索引（IVF_FLAT，聚类中心数100）
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(self.dim),  # L2距离度量
                self.dim,
                100,  # nlist（聚类中心数，建议为数据量的平方根）
                faiss.METRIC_L2
            )
            # 确保目录存在
            self.index_path.mkdir(parents=True, exist_ok=True)
            logger.info("创建新的Faiss索引")

    def save_index(self):
        """保存索引和元数据到本地（避免程序退出后丢失）"""
        if self.index is None or not self.metadata:
            logger.warning("无数据可保存")
            return

        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"索引保存完成，共{len(self.metadata)}条数据")

    def insert_documents(self, docs: List[Dict]) -> int:
        """
        插入文档到Faiss（需包含vector和metadata）
        :param docs: 文档列表，格式：[{"vector": [float], "text": str, "tech_tag": str, ...}]
        :return: 插入的文档数量
        """
        if not docs:
            return 0

        # 提取向量和元数据
        vectors = np.array([d["vector"] for d in docs], dtype=np.float32)
        new_metadata = [
            {k: v for k, v in d.items() if k != "vector"}  # 排除vector字段
            for d in docs
        ]

        # 首次插入需训练索引（IVF类型必需）
        if self.index.ntotal == 0:
            self.index.train(vectors)

        # 插入向量（Faiss会自动分配ID）
        self.index.add(vectors)
        # 追加元数据（与向量顺序严格对应）
        self.metadata.extend(new_metadata)

        # 保存索引（实时持久化）
        self.save_index()
        logger.info(f"插入{len(docs)}条文档，当前总数据量：{len(self.metadata)}")
        return len(docs)

    def search(self, query_vec: List[float], filters: Optional[Dict] = None, top_k: int = 10) -> List[Dict]:
        """
        检索相似向量并过滤元数据
        :param query_vec: 查询向量
        :param filters: 元数据过滤条件（如{"device_model": "S5720"}）
        :param top_k: 返回结果数量
        :return: 检索结果（含text、tech_tag、score等）
        """
        if self.index.ntotal == 0:
            return []

        # Faiss检索（返回距离和索引ID）
        query_np = np.array([query_vec], dtype=np.float32)
        distances, ids = self.index.search(query_np, top_k)

        # 转换结果（距离→相似度，关联元数据）
        results = []
        for i in range(len(ids[0])):
            doc_id = ids[0][i]
            distance = distances[0][i]
            # L2距离→相似度（归一化到0-1，值越大越相似）
            similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0

            # 获取元数据
            doc_meta = self.metadata[doc_id] if doc_id < len(self.metadata) else {}
            results.append({
                "doc_id": doc_id,
                "text": doc_meta.get("text", ""),
                "tech_tag": doc_meta.get("tech_tag", ""),
                "device_model": doc_meta.get("device_model", ""),
                "source": doc_meta.get("source", ""),
                "is_public": bool(doc_meta.get("is_public", True)),  ## 你筛选的字段，需要加上，返回来做判断的条件
                "score": similarity  # 相似度分数
            })

        # 应用元数据过滤（Faiss不支持原生过滤，需手动处理）
        if filters:
            filtered = []
            for res in results:
                match = True
                for key, value in filters.items():
                    # 支持单值或多值匹配（如{"tech_tag": ["VLAN", "OSPF"]}）
                    if isinstance(value, list) and res.get(key) not in value:
                        match = False
                        break
                    elif not isinstance(value, list) and res.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(res)
            results = filtered

        return results[:top_k]  # 确保返回数量正确
