from typing import List
from llama_index.core.schema import Node
from langchain.schema import Document
from .cleaner import TextCleaner

class LlamaToLangChainConverter:
    """
    将LlamaIndex的Node对象转换为LangChain的Document对象
    保留技术文档的元数据（设备型号、技术标签等）并适配格式
    """
    
    def __init__(self):
        self.cleaner = TextCleaner()
    
    def convert_single_node(self, node: Node) -> Document:
        """转换单个Node为LangChain Document"""
        # 清洗文本内容（处理格式、去重）
        cleaned_text = self.cleaner.clean(node.text)
        
        # 映射元数据（保留华为技术文档关键信息）
        langchain_metadata = {
            # 基础元数据
            "source": node.metadata.get("source", "unknown"),
            "doc_id": node.node_id,
            "page_label": node.metadata.get("page_label", ""),
            
            # 华为技术文档特有元数据
            "tech_tag": node.metadata.get("tech_tag", "unknown"),
            "device_model": node.metadata.get("device_model", "unknown"),
            "firmware_version": node.metadata.get("firmware_version", "unknown"),
            
            # 检索相关元数据
            "similarity_score": node.score if hasattr(node, "score") else 0.0
        }
        
        return Document(
            page_content=cleaned_text,
            metadata=langchain_metadata
        )
    
    def convert_nodes(self, nodes: List[Node]) -> List[Document]:
        """批量转换Node列表为LangChain Document列表"""
        return [self.convert_single_node(node) for node in nodes]
    
    def convert_with_filter(self, nodes: List[Node], min_score: float = 0.5) -> List[Document]:
        """
        转换并过滤低相似度结果
        :param min_score: 最低相似度分数（0-1），低于此值的文档将被过滤
        """
        filtered_docs = []
        for node in nodes:
            if hasattr(node, "score") and node.score < min_score:
                continue
            filtered_docs.append(self.convert_single_node(node))
        return filtered_docs
