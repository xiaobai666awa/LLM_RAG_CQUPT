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
                 classifier_model_path: str = r"app/modules/retrieval/model/bert_multiclass_final",
                 enable_category_filter: bool = False):  # 新增：分类过滤开关，默认关闭
        """
        混合检索器（支持分类过滤开关，默认关闭）
        :param enable_category_filter: 是否启用分类过滤（True=启用，False=关闭）
        """
        # 初始化嵌入模型
        self.embedder = SentenceTransformer("BAAI/bge-large-zh")

        # 初始化检索组件
        self.vector_client = FaissClient(index_path=faiss_index_path)
        self.bm25_index = BM25IndexManager(index_dir=r"app/data/bm25_index")
        self.reranker = CrossEncoderReranker()

        # 新增：根据开关决定是否加载分类模型
        self.enable_category_filter = enable_category_filter
        self.classifier_tokenizer = None
        self.classifier_model = None
        self.id2label = None

        if self.enable_category_filter:
            try:
                # 仅在启用时加载分类模型
                self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model_path)
                self.classifier_model = AutoModelForSequenceClassification.from_pretrained(
                    classifier_model_path
                )
                self.classifier_model.eval()
                self.id2label = self.classifier_model.config.id2label
                logger.info("分类过滤功能已启用，成功加载分类模型")
            except Exception as e:
                logger.error(f"启用分类过滤失败：分类模型加载错误{str(e)}", exc_info=True)
                self.enable_category_filter = False  # 加载失败时自动关闭
        else:
            logger.info("分类过滤功能默认关闭，不加载分类模型")

        # 权重校验
        if not (abs(vector_weight + bm25_weight - 1) < 1e-6):
            raise ValueError("向量权重+BM25权重必须为1")
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        # 加载元数据白名单（确保过滤条件有效）
        self.metadata_whitelist = self._load_metadata_whitelist()

    def _load_metadata_whitelist(self) -> Dict:
        """加载向量库中所有存在的元数据值（用于验证过滤条件）"""
        try:
            # 批量获取部分文档来提取元数据白名单
            sample_docs = self.vector_client.search(query_vec=np.zeros(1024), top_k=1000)
            if not sample_docs:
                return {"device_model": set(), "category": set(), "tech_tag": set()}

            # 提取所有可能的元数据值
            device_models = set(doc.get("device_model") for doc in sample_docs if doc.get("device_model"))
            categories = set(doc.get("category") for doc in sample_docs if doc.get("category"))
            tech_tags = set(tag for doc in sample_docs for tag in doc.get("tech_tag", []) if tag)

            return {
                "device_model": device_models,
                "category": categories,
                "tech_tag": tech_tags
            }
        except Exception as e:
            logger.warning(f"加载元数据白名单失败：{str(e)}，将跳过过滤条件验证")
            return {"device_model": set(), "category": set(), "tech_tag": set()}

    def _classify_query(self, query: str) -> str:
        """预测用户查询的分类（仅在启用分类过滤时执行）"""
        # 新增：未启用分类过滤时直接返回空
        if not self.enable_category_filter:
            logger.debug("分类过滤已关闭，跳过查询分类预测")
            return ""

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

            # 检查预测分类是否在白名单中
            if pred_category not in self.metadata_whitelist["category"]:
                logger.warning(f"预测分类{pred_category}不在知识库中，将不使用该分类过滤")
                return ""

            return pred_category
        except Exception as e:
            logger.warning(f"查询分类预测失败：{str(e)}，将跳过分类过滤")
            return ""

    def _extract_filters(self, query: str) -> Dict:
        """提取并验证过滤条件，确保有效性（分类过滤仅在启用时执行）"""
        filters = {}
        terms = extract_tech_terms(query)

        # 1. 提取设备型号（仅保留白名单中存在的型号）
        device_models = [t for t in terms if is_valid_device_model(t)]
        valid_devices = [d for d in device_models if d in self.metadata_whitelist["device_model"]]

        if valid_devices:
            filters["device_model"] = valid_devices[0]
            logger.info(f"提取到有效设备型号过滤条件：{valid_devices[0]}")
        else:
            if device_models:
                logger.warning(f"提取的设备型号{device_models}不在知识库中，跳过设备过滤")
            else:
                logger.info("未提取到设备型号，不添加设备过滤")

        # 2. 分类过滤（仅在启用且高置信度时使用）
        # 新增：未启用分类过滤时跳过
        if not self.enable_category_filter:
            logger.debug("分类过滤已关闭，不添加分类过滤条件")
            return filters

        pred_category = self._classify_query(query)
        if pred_category:
            with torch.no_grad():
                inputs = self.classifier_tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
                outputs = self.classifier_model(**inputs)
                pred_probs = torch.softmax(outputs.logits, dim=1)
                max_prob = pred_probs.max().item()

            if max_prob > 0.7:
                filters["category"] = pred_category
                logger.info(f"分类置信度{max_prob:.2f}，添加分类过滤：{pred_category}")
            else:
                logger.info(f"分类置信度{max_prob:.2f}（低于0.7），不添加分类过滤")

        return filters

    def _get_reliable_results(self, query_vec: np.ndarray, query: str, filters: Dict, top_k: int) -> tuple:
        """
        分级检索策略：确保在过滤条件过严时仍能返回结果
        1. 先使用完整过滤条件
        2. 若无结果，去除设备型号过滤
        3. 若仍无结果，去除所有过滤条件
        """
        # 1. 尝试完整过滤条件
        vector_results = self.vector_client.search(query_vec, filters, top_k)
        bm25_results = self.bm25_index.search(query, filters, top_k)

        if len(vector_results) + len(bm25_results) > 0:
            return vector_results, bm25_results, filters

        # 2. 去除设备型号过滤
        if "device_model" in filters:
            reduced_filters = {k: v for k, v in filters.items() if k != "device_model"}
            logger.info(f"完整过滤无结果，尝试去除设备型号过滤：{reduced_filters}")

            vector_results = self.vector_client.search(query_vec, reduced_filters, top_k)
            bm25_results = self.bm25_index.search(query, reduced_filters, top_k)

            if len(vector_results) + len(bm25_results) > 0:
                return vector_results, bm25_results, reduced_filters

        # 3. 去除所有过滤条件
        logger.warning(f"过滤条件过严，将返回所有相关结果（不进行过滤）")
        vector_results = self.vector_client.search(query_vec, {}, top_k)
        bm25_results = self.bm25_index.search(query, {}, top_k)

        return vector_results, bm25_results, {}

    def _merge_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """融合检索结果，保留URL和元数据"""

        # 归一化分数
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

        # 构建文档ID到结果的映射（保留所有元数据）
        doc_map = {}
        for i, res in enumerate(vector_results):
            doc_id = res["doc_id"]
            doc_map[doc_id] = {**res,
                               "merged_score": norm_vector[i] * self.vector_weight,
                               "source_url": res.get("source_url")}  # 保留URL

        for i, res in enumerate(bm25_results):
            doc_id = res["doc_id"]
            if doc_id in doc_map:
                doc_map[doc_id]["merged_score"] += norm_bm25[i] * self.bm25_weight
            else:
                doc_map[doc_id] = {**res,
                                   "merged_score": norm_bm25[i] * self.bm25_weight,
                                   "source_url": res.get("source_url")}  # 保留URL

        # 按融合分数排序
        merged = sorted(doc_map.values(), key=lambda x: x["merged_score"], reverse=True)[:20]
        logger.info(f"结果融合完成，向量{len(vector_results)}条+BM25{len(bm25_results)}条→去重后{len(merged)}条")
        return merged

    def retrieve(self, query: str, user_filters: Optional[Dict] = None, top_k: int = 5) -> List[Dict]:
        """执行混合检索，确保过滤可靠性和URL输出"""
        try:
            # 1. 提取并合并过滤条件
            auto_filters = self._extract_filters(query)
            filters = {**auto_filters, **(user_filters or {})}
            logger.info(f"初始检索过滤条件：{filters}")

            # 2. 生成查询向量
            query_vec = self.embedder.encode([query], convert_to_numpy=True)[0]
            query_vec = query_vec.astype(np.float32)

            # 3. 分级检索（确保有结果返回）
            vector_results, bm25_results, used_filters = self._get_reliable_results(
                query_vec=query_vec,
                query=query,
                filters=filters,
                top_k=10
            )
            logger.info(f"最终使用的过滤条件：{used_filters}")

            # 4. 融合检索结果
            merged_results = self._merge_results(vector_results, bm25_results)
            if not merged_results:
                logger.warning("所有检索策略均未返回结果")
                return []

            # 5. 重排序
            final_results = self.reranker.rerank(
                query=query,
                candidates=merged_results,
                top_k=top_k
            )

            # 6. 清洗结果（确保URL和元数据正确）
            for r in final_results:
                # 转换numpy类型为原生类型
                if "doc_id" in r and isinstance(r["doc_id"], np.integer):
                    r["doc_id"] = int(r["doc_id"])
                for k in ("score", "merged_score", "rerank_score"):
                    if k in r and r[k] is not None:
                        r[k] = float(r[k])

                # 确保URL字段存在且格式正确
                if "source_url" not in r:
                    r["source_url"] = None
                elif r["source_url"] == "unknown":
                    r["source_url"] = None

                # 处理技术标签格式
                if "tech_tag" in r and isinstance(r["tech_tag"], str):
                    r["tech_tag"] = r["tech_tag"].split(",") if r["tech_tag"] else []

                # 确保分类信息完整
                r["category"] = r.get("category", "unknown")

            logger.info(f"检索完成，返回{len(final_results)}条结果（含URL信息）")
            return final_results

        except Exception as e:
            logger.error(f"混合检索失败：{str(e)}", exc_info=True)
            raise RetrievalException(f"检索服务异常：{str(e)}")

