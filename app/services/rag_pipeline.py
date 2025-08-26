
from app.modules.retrieval import HybridRetriever
from typing import List, Dict
import app.core.model as bm
# import app.services.chat_logger as chat_logger
from collections import Counter  # 用于统计主要分类


class RAGPipeline:
    def __init__(self):
        # 初始化检索器（单例复用，避免重复加载模型）
        self.retriever = HybridRetriever(vector_weight=0.7, bm25_weight=0.3)
        # 定义分类专属回答模板（核心优化点）
        self.prompt_templates = {
            # 华为知识类：侧重概念解释、原理说明
            "huawei_knowledge": """
历史对话记录:
{history_text}

你是华为网络技术知识专家，负责解释网络技术概念和原理。

请基于以下技术文档内容回答用户问题，遵循以下规则：
1. 用通俗易懂的语言解释概念，避免过多命令行，必要时举例说明。
2. 结合网络场景说明技术用途（如「OSPF适用于大型企业网络的动态路由规划」）。
3. 若涉及华为特有术语，需补充全称和行业对应名称（如「VRP是华为通用路由平台操作系统，类似Cisco的IOS」）。
4. 最后用「来源：」标注引用的知识文档（格式：文档名称）。

文档内容：
{context_text}

用户问题：
{query}
回答：
    摘要：
            """,

            # 华为配置方案类：侧重步骤和命令
            "huawei_solutions": """
历史对话记录:
{history_text}

你是华为设备配置专家，负责提供交换机、路由器等设备的配置方案。

请基于以下技术文档内容回答用户问题，遵循以下规则：
1. 步骤清晰，使用编号列表，每步包含**核心命令**（加粗显示）和说明。
2. 明确标注命令适用的设备型号和固件版本（如「适用于S5720 V200R021及以上版本」）。
3. 包含配置验证命令（如「使用display vlan验证配置结果」）。
4. 若存在风险提示（如重启生效），需单独用「注意：」标注。
5. 最后用「来源：」标注引用的配置文档（格式：文档名称 - 章节）。**必须在回答末尾添加「参考文档」板块，包含文档标题和在线URL**，格式：  
   「{文档标题}：[{URL}]({URL})」

文档内容：
{context_text}

用户问题：
{query}
回答：
    摘要：
            """,

            # 华为RFC标准类：侧重标准原文和合规性
            "huawei_rfc": """
历史对话记录:
{history_text}

你是网络协议标准专家，负责解读RFC文档在华为设备中的实现。

请基于以下RFC文档内容回答用户问题，遵循以下规则：
1. 优先引用RFC原文描述（如「RFC 791第3.2节定义：IP地址长度为32位」）。
2. 说明华为设备对该RFC的支持情况（如「华为NE40E完全支持RFC 2547bis定义的MPLS VPN」）。
3. 对比标准要求与华为实现的差异（如有）。
4. 最后用「来源：」标注引用的RFC编号和华为文档（格式：RFC XXXX，华为文档名称）。

文档内容：
{context_text}

用户问题：
{query}
回答：
    摘要：
            """,

            # 思科配置类：使用思科命令体系
            "cisco_config": """
历史对话记录:
{history_text}

你是思科设备配置专家，负责提供交换机、路由器等设备的配置方案。

请基于以下技术文档内容回答用户问题，遵循以下规则：
1. 使用思科命令体系（如「enable → configure terminal」），核心命令加粗显示。
2. 明确标注适用的设备型号（如「适用于Catalyst 9300系列交换机」）。
3. 步骤包含进入特权模式、全局配置模式等必要前置操作。
4. 提供配置验证命令（如「show vlan brief」）。
5. 最后用「来源：」标注引用的思科文档（格式：文档名称 - 章节）。

文档内容：
{context_text}

用户问题：
{query}
回答：
    摘要：
            """,

            # 默认模板：无法识别分类时使用
            "default": """
历史对话记录:
{history_text}

你是网络技术专家，负责解答网络相关问题。

请基于以下文档内容回答用户问题，确保信息准确、步骤清晰。

文档内容：
{context_text}

用户问题：
{query}
回答：
    摘要：
            """
        }

    def retrieve_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """单独获取检索结果（含分类信息）"""
        return self.retriever.retrieve(query, top_k=top_k)

    def _get_main_category(self, retrieved_docs: List[Dict]) -> str:
        """获取检索结果的主要分类（出现次数最多的分类）"""
        if not retrieved_docs:
            return "default"
        # 提取所有文档的分类
        categories = [doc.get("category", "default") for doc in retrieved_docs]
        # 统计出现次数最多的分类
        main_category = Counter(categories).most_common(1)[0][0]
        # chat_logger.info(f"检索结果主要分类：{main_category}（基于{len(retrieved_docs)}条文档）")
        return main_category

    def build_rag_prompt(self, query: str, history: List[Dict[str, str]], model: dict) -> str:
        """根据文档分类动态生成Prompt"""
        # 1. 获取检索结果（含分类信息）
        print("开始查询rag库")
        retrieved_docs = self.retrieve_docs(query, top_k=5)
        print("retrieved_docs:", retrieved_docs)

        # 2. 提取上下文和主要分类
        context_text = "\n".join([doc.get("text", "") for doc in retrieved_docs])
        main_category = self._get_main_category(retrieved_docs)

        # 3. 拼接历史对话
        history_text = ""
        for turn in history:
            history_text += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"

        # 4. 选择对应分类的Prompt模板
        prompt_template = self.prompt_templates.get(main_category, self.prompt_templates["default"])

        # 5. 填充模板变量
        prompt = prompt_template.format(
            history_text=history_text.strip(),
            context_text=context_text,
            query=query
        )

        # 6. 调用模型生成回答
        print("历史对话：", history_text.strip())
        md = bm.get_chat_model(type=model['type'], name=model['name'])
        print("模型实例类型：", type(md))
        return md.invoke(prompt).content


if __name__ == "__main__":
    # 测试不同分类的回答效果
    rag = RAGPipeline()

    # # 测试1：华为ip问题
    # print("=== 测试华为配置类问题 ===")
    # print(rag.build_rag_prompt(
    #     query="为什么需要5G-R？",
    #     history=[],
    #     model={"type": "siliconflow", "name": "deepseek-ai/DeepSeek-V3"}
    # ))
    #
    # # 测试2：思科配置类问题
    # print("\n=== 测试思科配置类问题 ===")
    # print(rag.build_rag_prompt(
    #     query="ecore9300 TTU业务出现批量去附着故障怎么办？",
    #     history=[],
    #     model={"type": "siliconflow", "name": "deepseek-ai/DeepSeek-V3"}
    # ))
    #
    # # 测试3：知识类问题
    # print("\n=== 测试知识类问题 ===")
    # print(rag.build_rag_prompt(
    #     query="什么是OSPF协议？",
    #     history=[],
    #     model={"type": "siliconflow", "name": "deepseek-ai/DeepSeek-V3"}
    # ))