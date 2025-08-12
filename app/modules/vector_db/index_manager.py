from .client import FaissClient
from app.common.logger import logger
from typing import Dict

class FaissIndexManager:
    """Faiss索引管理器（功能简化，因为Faiss索引在创建时已定义）"""

    def __init__(self, client: FaissClient):
        self.client = client

    # def optimize(self):
    #     """优化索引（针对IVF类型，重建聚类中心，提升检索速度）"""
    #     if self.client.index.is_trained:
    #         self.client.index.rebuild_index()
    #         logger.info("Faiss索引优化完成（重建聚类中心）")
    #     else:
    #         logger.warning("索引未训练，无法优化")

    def optimize(self):
        """优化索引（IVF类型无需显式优化，默认已训练）"""
        if self.client.index.is_trained:
            # IVF 类型的索引在初始 train 后已就绪，通常不需要再优化
            self.client.save_index()  # 可选：保存当前状态
            logger.info("Faiss索引已训练，无需重建，已保存索引")
        else:
            logger.warning("索引未训练，无法优化")
    def stats(self) -> Dict:
        """返回索引统计信息"""
        return {
            "total_docs": self.client.index.ntotal,
            "index_type": type(self.client.index).__name__,
            "dimension": self.client.dim
        }
