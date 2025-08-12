from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, KEYWORD, BOOLEAN
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.scoring import BM25F
from pathlib import Path
from app.common.logger import logger
from app.common.constants import TechTag, DeviceModel

import jieba
from whoosh.analysis import Token

class ChineseAnalyzer:


    """简易中文分词器（支持华为技术术语，兼容Whoosh analyzer协议）"""
    def __call__(self, text, **kwargs):
        # kwargs 用来接收 whoosh 传入的 mode / positions / chars 等
        jieba.add_word("S5720")  # 华为设备型号
        jieba.add_word("vlan")  # 技术术语
        jieba.add_word("system-view")  # 命令

        t = Token(**kwargs)  # 创建Token模板
        pos = 0
        start = 0
        for w in jieba.cut(text, cut_all=False):
            w = w.strip()
            if not w:
                continue
            t.text = w
            t.pos = pos
            t.startchar = start
            t.endchar = start + len(w)
            pos += 1
            start = t.endchar
            yield t

# 华为技术文档的BM25索引schema（优化技术术语检索）
HUAWEI_SCHEMA = Schema(
    doc_id=ID(stored=True, unique=True),  # 文档唯一ID
    text=TEXT(stored=True, analyzer=ChineseAnalyzer()),  # 全文内容（中文分词）
    tech_tag=KEYWORD(stored=True, commas=True),  # 技术标签（如"交换机配置,VLAN"）
    device_model=KEYWORD(stored=True),  # 设备型号（如"S5720,S5730"）
    is_public=BOOLEAN(stored=True)  # 是否公开文档
)

class BM25IndexManager:
    def __init__(self, index_dir: str = "data/bm25_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        if exists_in(str(self.index_dir)):
            logger.info(f"加载BM25索引：{self.index_dir}")
            return open_dir(str(self.index_dir))
        else:
            logger.info(f"创建新BM25索引：{self.index_dir}")
            return create_in(str(self.index_dir), HUAWEI_SCHEMA)

    def add_documents(self, documents: list):
        """
        添加文档到BM25索引
        :param documents: 文档列表，每个文档含doc_id/text/tech_tag/device_model/is_public
        """
        writer = self.index.writer()
        for doc in documents:
            # 技术标签转换为逗号分隔字符串（适配KEYWORD类型）
            tech_tags = ",".join(doc["tech_tag"]) if isinstance(doc["tech_tag"], list) else doc["tech_tag"]
            writer.add_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                tech_tag=tech_tags,
                device_model=doc["device_model"],
                is_public=doc["is_public"]
            )
        writer.commit()
        logger.info(f"BM25索引添加{len(documents)}条文档")

    def search(self, query: str, filters: dict = None, top_k: int = 10) -> list:
        """
        BM25关键词检索
        :param query: 检索关键词（如"VLAN配置步骤"）
        :param filters: 过滤条件（如{"device_model": "S5720", "is_public": True}）
        :param top_k: 返回结果数量
        :return: 检索结果列表，含doc_id/text/score等
        """
        results = []
        with self.index.searcher(weighting=BM25F(B=0.75, K1=1.2)) as searcher:
            # 多字段检索（优先匹配text，其次tech_tag）
            parser = MultifieldParser(
                ["text", "tech_tag"],
                schema=self.index.schema,

            )
            whoosh_query = parser.parse(query)

            # 应用过滤条件（如设备型号）
            if filters:
                for key, value in filters.items():
                    if key == "device_model":
                        whoosh_query = whoosh_query & QueryParser("device_model", self.index.schema).parse(value)
                    elif key == "is_public":
                        whoosh_query = whoosh_query & QueryParser("is_public", self.index.schema).parse(str(value).lower())

            # 执行检索
            hits = searcher.search(whoosh_query, limit=top_k)
            for hit in hits:
                results.append({
                    "doc_id": hit["doc_id"],
                    "text": hit["text"],
                    "tech_tag": hit["tech_tag"].split(","),
                    "device_model": hit["device_model"],
                    "is_public": bool(hit.get("is_public", True)),
                    "score": hit.score  # BM25分数
                })
        logger.info(f"BM25检索完成，返回{len(results)}条结果")
        return results
