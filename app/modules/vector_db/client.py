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
        self.index_path = Path(index_path).resolve()  # 转为绝对路径，避免相对路径混乱
        self.dim = dim
        self.index = None  # Faiss索引对象
        self.metadata = []  # 存储文档元数据（与向量一一对应）
        self._load_or_create_index()

    def _load_or_create_index(self):
        """加载已有索引或创建新索引（优化：动态调整聚类参数，增强兼容性）"""
        # 索引文件路径标准化
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.pkl"

        try:
            if index_file.exists() and meta_file.exists():
                # 加载已有索引（增加完整性校验）
                self.index = faiss.read_index(str(index_file))
                with open(meta_file, "rb") as f:
                    self.metadata = pickle.load(f)

                # 校验向量数与元数据数是否一致（核心修复点）
                if len(self.metadata) != self.index.ntotal:
                    raise RetrievalException(
                        f"索引与元数据数量不匹配（向量：{self.index.ntotal}, 元数据：{len(self.metadata)}）"
                    )
                logger.info(f"加载Faiss索引成功，共{len(self.metadata)}条数据（维度：{self.dim}）")

            else:
                # 创建新索引（优化：根据预期数据量动态调整聚类数）
                # 当数据量<1000时使用Flat索引，避免IVF训练警告
                self.index = faiss.IndexFlatL2(self.dim) if self.dim <= 2048 else faiss.IndexFlatIP(self.dim)
                self.index_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建新索引（{type(self.index).__name__}，维度：{self.dim}）")

        except Exception as e:
            logger.error(f"索引初始化失败：{str(e)}", exc_info=True)
            # 索引损坏时重建
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadata = []
            logger.warning("已自动重建空索引")

    def save_index(self):
        """保存索引和元数据（优化：元数据字段校验）"""
        if self.index is None or not self.metadata:
            logger.warning("无有效数据可保存")
            return

        try:
            # 保存前校验关键元数据字段（确保category等必要字段存在）
            missing_fields = set()
            for i, meta in enumerate(self.metadata[:5]):  # 抽样检查前5条
                for field in ["category", "device_model", "tech_tag"]:
                    if field not in meta:
                        missing_fields.add(f"文档{i}缺少{field}")

            if missing_fields:
                logger.warning(f"元数据字段不完整：{', '.join(missing_fields)}")

            # 保存索引
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
            # 保存元数据（指定pickle协议，增强兼容性）
            with open(self.index_path / "metadata.pkl", "wb") as f:
                pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"索引保存完成，共{len(self.metadata)}条数据（向量数：{self.index.ntotal}）")

        except Exception as e:
            logger.error(f"索引保存失败：{str(e)}", exc_info=True)
            raise RetrievalException(f"保存索引时出错：{str(e)}")

    def insert_documents(self, docs: List[Dict]) -> int:
        """
        插入文档（优化：强制保留关键元数据字段，增强类型校验）
        :param docs: 文档列表，必须包含"vector"和关键元数据（category等）
        :return: 插入数量
        """
        if not docs:
            return 0

        try:
            # 1. 向量提取与校验
            vectors = []
            for doc in docs:
                # 校验向量存在且维度正确
                if "vector" not in doc or len(doc["vector"]) != self.dim:
                    raise ValueError(f"向量格式错误（维度应为{self.dim}，实际{len(doc.get('vector', []))}）")
                vectors.append(doc["vector"])

            vectors_np = np.array(vectors, dtype=np.float32)

            # 2. 元数据处理（强制保留关键字段，避免丢失）
            new_metadata = []
            for doc in docs:
                # 提取元数据，确保包含关键字段（缺失则补unknown）
                meta = {
                    "category": doc.get("category", "unknown"),
                    "device_model": doc.get("device_model", "unknown"),
                    "tech_tag": doc.get("tech_tag", "unknown"),
                    "text": doc.get("text", ""),
                    "source": doc.get("source", ""),
                    "is_public": doc.get("is_public", True)
                }
                # 保留其他额外字段
                for k, v in doc.items():
                    if k not in meta and k != "vector":
                        meta[k] = v
                new_metadata.append(meta)

            # 3. 插入向量（根据索引类型处理训练）
            if hasattr(self.index, "is_trained") and not self.index.is_trained:
                # 仅IVF类索引需要训练（自动判断数据量）
                train_size = min(10000, len(vectors_np))  # 最多用10k样本训练
                self.index.train(vectors_np[:train_size])
                logger.info(f"索引训练完成（使用{train_size}个样本）")

            self.index.add(vectors_np)
            self.metadata.extend(new_metadata)

            # 4. 保存并返回结果
            self.save_index()
            inserted = len(docs)
            logger.info(f"插入{inserted}条文档，当前总数据量：{len(self.metadata)}")
            return inserted

        except Exception as e:
            logger.error(f"插入文档失败：{str(e)}", exc_info=True)
            raise RetrievalException(f"插入文档时出错：{str(e)}")

    def search(self, query_vec: List[float], filters: Optional[Dict] = None, top_k: int = 10) -> List[Dict]:
        """
        检索优化：增强边界检查，确保元数据完整返回
        """
        if self.index.ntotal == 0:
            logger.warning("索引为空，返回空结果")
            return []

        try:
            # 校验查询向量维度
            if len(query_vec) != self.dim:
                raise ValueError(f"查询向量维度错误（应为{self.dim}，实际{len(query_vec)}）")

            # 向量检索
            query_np = np.array([query_vec], dtype=np.float32)
            distances, ids = self.index.search(query_np, min(top_k * 2, self.index.ntotal))  # 多取一倍用于过滤

            # 组装结果（严格边界检查）
            results = []
            for i in range(len(ids[0])):
                doc_id = ids[0][i]
                # 防止索引越界（核心修复点）
                if doc_id < 0 or doc_id >= len(self.metadata):
                    logger.warning(f"无效文档ID：{doc_id}（总元数据数：{len(self.metadata)}）")
                    continue

                # 计算相似度并关联元数据
                distance = distances[0][i]
                doc_meta = self.metadata[doc_id]
                results.append({
                    "doc_id": doc_id,
                    "score": 1.0 / (1.0 + distance) if distance > 0 else 1.0,  # L2转相似度
                    **doc_meta  # 展开所有元数据字段（确保category等被返回）
                })

            # 应用过滤（支持多值匹配和模糊匹配）
            if filters:
                results = self._apply_filters(results, filters)

            # 按分数排序并限制数量
            return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        except Exception as e:
            logger.error(f"检索失败：{str(e)}", exc_info=True)
            raise RetrievalException(f"检索时出错：{str(e)}")

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """优化过滤逻辑：支持模糊匹配和默认值处理"""
        filtered = []
        for res in results:
            match = True
            for key, value in filters.items():
                # 获取元数据值（默认unknown，避免KeyError）
                res_val = res.get(key, "unknown")

                # 多值匹配（如{"category": ["huawei_knowledge", "cisco_config"]}）
                if isinstance(value, list):
                    if res_val not in value:
                        match = False
                        break
                # 单值匹配（支持模糊匹配，如型号包含关键词）
                elif isinstance(value, str):
                    if value not in res_val and res_val != value:
                        match = False
                        break
            if match:
                filtered.append(res)
        return filtered
