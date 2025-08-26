from app.modules.vector_db import FaissClient
from .bm25_index import BM25IndexManager
from .reranker import CrossEncoderReranker
from app.common.logger import logger
from app.common.utils import extract_tech_terms, is_valid_device_model
from app.common.exceptions import RetrievalException
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch


class HybridRetriever:
    def __init__(self, vector_weight: float = 0.7, bm25_weight: float = 0.3,
                 faiss_index_path: str = r"app/data/faiss_index",
                 classifier_model_path: str = r"app/modules/retrieval/model/bert_multiclass_final"):
        """
        混合检索器（新增分类预测功能）
        :param vector_weight: 向量检索权重
        :param bm25_weight: BM25检索权重
        :param classifier_model_path: BERT分类模型路径（预测查询分类）
        """
        # 初始化嵌入模型
        self.embedder = SentenceTransformer("BAAI/bge-large-zh")

        # 初始化检索组件
        self.vector_client = FaissClient(index_path=faiss_index_path)
        self.bm25_index = BM25IndexManager(index_dir=r"app/data/bm25_index")
        self.reranker = CrossEncoderReranker()

        # 初始化分类模型（预测查询所属分类）
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
        self.classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model_path)
        self.classifier_model.eval()  # 推理模式
        self.id2label = self.classifier_model.config.id2label  # 标签映射（id→分类名）

        # 权重校验
        if not (abs(vector_weight + bm25_weight - 1) < 1e-6):
            raise ValueError("向量权重+BM25权重必须为1")
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def _classify_query(self, query: str) -> str:
        """预测用户查询的分类（如huawei_solutions/cisco_config）"""
        try:
            inputs = self.classifier_tokenizer(
                query,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self.classifier_model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            pred_category = self.id2label[pred_id]
            logger.info(f"查询分类预测：{query} → {pred_category}")
            return pred_category
        except Exception as e:
            logger.warning(f"查询分类预测失败：{str(e)}，将跳过分类过滤")
            return ""

    def _extract_filters(self, query: str) -> Dict:
        """从查询中提取过滤条件（新增分类过滤）"""
        filters = {}
        terms = extract_tech_terms(query)

        # 1. 提取设备型号（如"S5720"）
        device_models = [t for t in terms if is_valid_device_model(t)]
        if device_models:
            filters["device_model"] = device_models[0]  # 取第一个匹配的型号

        # 2. 提取技术标签（如"VLAN"→映射到"交换机配置-VLAN"）
        tech_tags = [t for t in terms if any(t in tag for tag in ["VLAN", "OSPF", "路由", "交换机"])]
        if tech_tags:
            filters["tech_tag"] = tech_tags

        # 3. 新增：添加分类过滤（基于BERT预测结果）
        pred_category = self._classify_query(query)
        if pred_category:
            filters["category"] = pred_category  # 核心：按预测分类过滤

        return filters

    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """融合向量检索和BM25检索结果（保留分类信息）"""

        # 归一化分数（0-1范围）
        def normalize(scores):
            if not scores:
                return []
            min_s = min(scores)
            max_s = max(scores)
            return [(s - min_s) / (max_s - min_s) if max_s != min_s else 0.5 for s in scores]

        # 归一化向量和BM25分数
        vector_scores = [r["score"] for r in vector_results]
        bm25_scores = [r["score"] for r in bm25_results]
        norm_vector = normalize(vector_scores)
        norm_bm25 = normalize(bm25_scores)

        # 构建文档ID到结果的映射（去重，保留分类信息）
        doc_map = {}
        for i, res in enumerate(vector_results):
            doc_id = res["doc_id"]
            # 确保保留分类信息
            doc_map[doc_id] = {**res,
                               "merged_score": norm_vector[i] * self.vector_weight,
                               "category": res.get("category", "unknown")}  # 显式保留分类

        for i, res in enumerate(bm25_results):
            doc_id = res["doc_id"]
            if doc_id in doc_map:
                # 已存在，叠加BM25权重
                doc_map[doc_id]["merged_score"] += norm_bm25[i] * self.bm25_weight
            else:
                # 新文档，直接计算融合分数
                doc_map[doc_id] = {**res,
                                   "merged_score": norm_bm25[i] * self.bm25_weight,
                                   "category": res.get("category", "unknown")}  # 显式保留分类

        # 按融合分数排序，取前20条（为重排序保留足够候选）
        merged = sorted(doc_map.values(), key=lambda x: x["merged_score"], reverse=True)[:20]
        logger.info(f"结果融合完成，向量{len(vector_results)}条+BM25{len(bm25_results)}条→去重后{len(merged)}条")
        return merged

    def retrieve(self, query: str, user_filters: Optional[Dict] = None, top_k: int = 5) -> List[Dict]:
        """
        执行混合检索（新增分类过滤，返回结果含分类信息）
        :param query: 用户查询
        :param user_filters: 用户指定过滤条件（支持{"category": "cisco_config"}强制筛选）
        :param top_k: 最终返回数量
        :return: 检索结果（每条含category字段标注来源分类）
        """
        try:
            # 1. 提取自动过滤条件（含分类预测）
            auto_filters = self._extract_filters(query)
            # 合并用户过滤条件（用户指定优先，可覆盖自动分类）
            filters = {**auto_filters, **(user_filters or {})}
            logger.info(f"最终检索过滤条件：{filters}")

            # 2. 向量检索
            query_vec = self.embedder.encode([query], convert_to_numpy=True)[0]
            query_vec = query_vec.astype(np.float32)
            vector_results = self.vector_client.search(
                query_vec=query_vec,
                filters=filters,
                top_k=10
            )

            # 3. BM25关键词检索
            bm25_results = self.bm25_index.search(
                query=query,
                filters=filters,
                top_k=10
            )

            # 4. 融合检索结果（保留分类信息）
            merged_results = self._merge_results(vector_results, bm25_results)
            if not merged_results:
                logger.warning("未检索到任何结果")
                return []

            # 5. 重排序（提升相关性）
            final_results = self.reranker.rerank(
                query=query,
                candidates=merged_results,
                top_k=top_k
            )

            # 6. 清洗结果（确保分类信息完整）
            for r in final_results:
                # 转换numpy类型为原生类型
                if "doc_id" in r and isinstance(r["doc_id"], np.integer):
                    r["doc_id"] = int(r["doc_id"])
                for k in ("score", "merged_score", "rerank_score"):
                    if k in r and r[k] is not None:
                        r[k] = float(r[k])
                # 确保tech_tag格式统一
                if "tech_tag" in r and isinstance(r["tech_tag"], str):
                    r["tech_tag"] = r["tech_tag"].split(",") if r["tech_tag"] else []
                # 确保category字段存在（标注来源分类）
                if "category" not in r:
                    r["category"] = "unknown"

            logger.info(f"检索完成，返回{len(final_results)}条结果（含分类信息）")
            return final_results

        except Exception as e:
            logger.error(f"混合检索失败：{str(e)}", exc_info=True)
            raise RetrievalException(f"检索服务异常：{str(e)}")

