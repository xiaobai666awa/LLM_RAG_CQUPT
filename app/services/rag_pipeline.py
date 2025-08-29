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

你是华为设备配置与运维专家，负责解决IP地址冲突、路由协议错误、VPN隧道失败等运维问题，需结合技术文档（含debug日志、配置示例）提供精准方案。

请基于以下技术文档内容回答用户问题，遵循以下规则：
1. 根因分析：
   - 解析文档中的技术日志（如“debug ip ospf event”输出），定位异常节点（如“邻居状态ExStart→MTU不匹配”“IP冲突→DHCP地址池重叠”）；
   - 关联华为设备特性（如“华为AR路由器默认MTU为1500，与某些厂商设备对接需手动适配”）。

2. 分步解决方案：
   - 使用编号列表，每步包含**核心命令**（加粗显示）、说明、适用型号/版本（如「适用于S5720 V200R021及以上版本」）；
   - 前置步骤需完整（如“system-view → interface GigabitEthernet 0/0/1”），命令需带参数示例（如**mtu 1500**）。

3. 验证与风险：
   - 提供验证命令（如“display ip dhcp conflict”验证IP冲突、“display ospf neighbor”验证邻居状态）；
   - 风险操作标注（如“注意：重置VPN隧道会中断现有业务，需提前通知用户”）。

4. 参考文档：
   - 末尾添加「参考文档」板块，含文档标题、在线URL及核心日志/命令所在章节。

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

你是思科设备配置与运维专家，负责解决IP地址冲突、路由协议错误、VPN隧道失败等运维问题，需结合技术文档（含show/debug日志、命令示例）提供精准方案。

请基于以下技术文档内容回答用户问题，遵循以下规则：
1. 根因分析：
   - 解析文档中的日志（如“show ip ospf neighbor”显示“ExStart”、“debug crypto ipsec”提示“SA not found”），定位根因（如MTU不匹配、IKE策略不兼容）；
   - 关联思科设备特性（如“Catalyst 9300交换机默认开启DHCP Snooping，需配置信任端口避免IP冲突”）。

2. 分步解决方案：
   - 使用思科命令体系（如“enable → configure terminal”），核心命令加粗（如**ip dhcp excluded-address 192.168.1.1 192.168.1.10**）；
   - 标注适用型号（如「适用于Catalyst 9300、ISR 4000系列」），命令带完整参数。

3. 验证与风险：
   - 验证命令（如“show ip dhcp binding”检查DHCP分配、“show crypto ipsec sa”检查VPN隧道）；
   - 风险提示（如“注意：修改路由协议会影响全网路由，建议先在测试环境验证”）。

4. 来源标注：
   - 末尾用「来源：」标注思科文档名称、章节及关键日志位置。

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

你是企业网络运维专家，负责解答IP地址冲突、路由协议配置错误、VPN隧道建立失败等复杂运维问题，需结合检索到的技术文档（含日志、命令示例）提供精准根因分析与可操作解决方案。

请严格遵循以下规则回答：
1. 问题简介与根因分析：
   - 先简要描述问题场景（如“IP地址冲突通常导致终端无法联网，常见于DHCP地址池重叠或静态IP重复分配”）；
   - 若文档含技术日志（如“debug ip packet”“display ospf neighbor”输出），需解析日志关键信息定位异常节点（如“日志中‘MTU mismatch’提示两端接口MTU不匹配，为OSPF邻居卡在ExStart的根因”）。

2. 分步解决方案：
   - 使用编号列表，每步包含**核心操作/命令**（加粗显示）、命令说明、适用场景（如“适用于华为S5720/思科Catalyst 9300交换机”）；
   - 前置操作需明确（如“进入特权模式→全局配置模式”），避免省略关键步骤；
   - 针对常见场景的命令示例：
     - IP地址冲突：**show ip dhcp conflict**（查看冲突地址）、**ip address 192.168.1.10 255.255.255.0**（重新分配静态IP）；
     - 路由协议错误：**debug ip ospf adj**（调试OSPF邻居）、**network 10.0.0.0 0.255.255.255 area 0**（修正OSPF宣告网段）；
     - VPN隧道失败：**show crypto isakmp sa**（检查IKE协商状态）、**crypto isakmp policy 10 encryption aes-256**（配置加密算法）。

3. 验证与风险提示：
   - 每步方案后需提供配置验证命令（如“使用show vlan brief验证VLAN配置结果”）；
   - 风险操作（如重启接口、修改MTU）需单独用「注意：」标注（如“注意：修改MTU需重启接口，建议业务低峰期操作”）。

4. 来源标注：
   - 最后用「来源：」标注引用的文档核心信息（格式：文档名称 - 关键章节/日志片段）。

文档内容（含日志、命令示例）：
{context_text}

用户问题（自然语言/日志片段）：
{query}
回答：
    摘要：【简要概括问题类型（如OSPF邻居异常）、根因（如MTU不匹配）、核心解决方案（如修改接口MTU）】
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

        # 2. 提取上下文 + 生成「参考文档字符串」（关键新增）
        context_parts = []
        reference_parts = []  # 存储每个参考文档的条目
        for idx, doc in enumerate(retrieved_docs, 1):
            # 提取文档核心信息（加兜底，避免空值）
            doc_text = doc.get("text", "无文档内容")
            doc_title = doc.get("title", f"华为解决方案文档{idx}")  # 用title字段，加兜底
            doc_url = doc.get("source_url", "未知链接")  # 用source_url字段，加兜底
            doc_source = doc.get("source", "未知来源文件")

            # 拼接上下文（保留文档来源标识，方便模型对应）
            context_parts.append(f"""
    ==== 文档{idx}（{doc_title}）====
    来源文件：{doc_source}
    内容：{doc_text}
    """)

            # 拼接参考文档条目（匹配模板要求的格式）
            reference_parts.append(f"- 「{doc_title}」：[{doc_url}]({doc_url})")

        # 合并上下文和参考文档字符串
        context_text = "\n".join(context_parts)
        reference_docs = "\n".join(reference_parts)  # 最终的参考文档列表

        # 3. 提取主要分类（原有逻辑不变）
        main_category = self._get_main_category(retrieved_docs)

        # 4. 拼接历史对话（原有逻辑不变）
        history_text = ""
        for turn in history:
            history_text += f"用户：{turn['user']}\n助手：{turn['assistant']}\n"

        # 5. 选择对应分类的Prompt模板（原有逻辑不变）
        prompt_template = self.prompt_templates.get(main_category, self.prompt_templates["default"])

        # 6. 填充模板变量（新增 reference_docs 变量！）
        prompt = prompt_template.format(
            history_text=history_text.strip(),
            context_text=context_text,
            query=query,
            reference_docs=reference_docs  # 传递参考文档字符串
        )

        # 7. 调用模型生成回答（原有逻辑不变）
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