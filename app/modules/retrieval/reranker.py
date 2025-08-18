from sentence_transformers import CrossEncoder
from app.common.logger import logger
from typing import List, Dict

class CrossEncoderReranker:
    # def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cpu"):
    def __init__(self, model_name: str = r"BAAI/bge-reranker-large", device: str = "cpu"):
        """
        初始化重排序模型
        :param model_name: 重排序模型（BGE重排序模型对中文技术文档友好）
        :param device: 运行设备（cpu/gpu）
        """
        # self.model = CrossEncoder(model_name, device=device)
        # logger.info(f"加载重排序模型：{model_name}（设备：{device}）")
        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"加载重排序模型：{model_name}（设备：{device}）")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        对检索候选结果重排序
        :param query: 用户查询
        :param candidates: 候选结果列表（含text字段）
        :param top_k: 重排序后返回的结果数量
        :return: 按相关性排序的结果（添加rerank_score字段）
        """
        if not candidates:
            return []
        
        # 构造（query, text）对
        pairs = [(query, candidate["text"]) for candidate in candidates]
        
        # 计算相关性分数（越高越相关）
        scores = self.model.predict(pairs)
        
        # 关联分数并排序
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        
        # 按rerank_score降序排列，取top_k
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        logger.info(f"重排序完成，原始{len(candidates)}条→保留{len(reranked)}条")
        return reranked
