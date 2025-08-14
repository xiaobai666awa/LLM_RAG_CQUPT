from app.modules.retrieval import HybridRetriever
from typing import List, Dict

class RAGPipeline:


    def build_rag_prompt(self,query: str, history: List[Dict[str, str]]):
        retriever = HybridRetriever(vector_weight=0.7, bm25_weight=0.3)
        retrieved_docs=retriever.retrieve(query)
        # 检索结果拼成 context
        context_text = "\n".join([doc.get("text", "") for doc in retrieved_docs])
        # 历史对话拼接
        history_text = ""
        for turn in history:
            role = "用户" if turn["role"] == "user" else "助手"
            history_text += f"{role}：{turn['content']}\n"

        # 最终 prompt
        prompt = f"""
你是华为网络技术专家，负责解答交换机、路由器等设备的配置与故障处理问题。

请基于以下技术文档内容回答用户问题，遵循以下规则：
1. 优先使用文档中的官方命令和步骤，确保准确性（如华为S5720的VLAN配置命令为`vlan batch <ID>`）。
2. 明确标注命令适用的设备型号和固件版本（如「适用于S5720 V200R021及以上版本」）。
3. 步骤清晰，使用编号列表，关键命令单独成行并加粗（如**system-view**）。
4. 若文档中存在多个解决方案，按「推荐方案→备选方案」排序。
5. 最后用「来源：」标注引用的文档（格式：文档名称 - 章节），多个来源用逗号分隔。

文档内容：
{context_text}

用户问题：
{query}
回答：
    
    摘要：
    """
        return prompt
