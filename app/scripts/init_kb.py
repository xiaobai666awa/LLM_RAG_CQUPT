import json
from pathlib import Path
from collections import defaultdict
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

from app.core.model import get_embedding_model
from app.modules.vector_db.client import FaissClient
from app.modules.vector_db.index_manager import FaissIndexManager
from app.common.logger import logger
from app.common.utils import GPT4VOCRProcessor  # 引入OCR工具（处理图片/PDF）

# -------------------------- 核心配置（基础规则） --------------------------
# 1. 分类映射：子目录名 → 分类标签（需与DATA目录结构对应）
CATEGORY_MAP = {
    "knowledge": "huawei_knowledge",  # 华为-基础知识
    "RFC": "huawei_rfc",  # 华为-RFC文档
    "solutions": "huawei_solutions",  # 华为-配置方案
    "cisco": "cisco_config"  # 思科-配置手册
}

# 2. 基础设备型号规则（分厂商）- 静态基础规则
BASE_DEVICE_MODEL_RULES = {
    "huawei": ["S5720", "S12700", "NE40E", "AR650"],  # 华为设备型号
    "cisco": ["Catalyst 9300", "ISR 4000", "Nexus 9000"]  # 思科设备型号
}

# 3. 基础技术标签规则 - 静态基础规则
BASE_TECH_TAG_RULES = {
    "vlan": "交换机配置-VLAN",
    "ospf": "路由协议-OSPF",
    "bgp": "路由协议-BGP",
    "acl": "安全策略-ACL",
    "nat": "地址转换-NAT"
}


# -------------------------- 动态规则扩展 --------------------------
def collect_dynamic_rules(docs: list) -> tuple:
    """从文档中动态收集设备型号和技术标签规则"""
    # 动态收集设备型号（按厂商）
    dynamic_device_models = defaultdict(set)
    # 动态收集技术标签
    dynamic_tech_tags = set()

    for doc in docs:
        # 从文件名和内容中提取可能的设备型号
        file_name = doc.metadata.get("file_name", "").lower()
        content = doc.text.lower()

        # 识别厂商
        category = doc.metadata.get("category", "")
        vendor = "huawei" if category.startswith("huawei") else "cisco"

        # 提取潜在设备型号（长度4-10的大写字母+数字组合）
        import re
        potential_models = re.findall(r'[A-Z]+\d+[A-Z]*\d*', file_name + " " + content)
        for model in potential_models:
            if 4 <= len(model) <= 10:  # 过滤过短或过长的型号
                dynamic_device_models[vendor].add(model)

        # 提取潜在技术标签（从内容中高频技术术语）
        potential_tags = re.findall(r'\b(?:[a-zA-Z0-9_-]+)\b', content)
        # 筛选可能的技术标签（长度3-20，非普通词汇）
        common_words = {"the", "and", "for", "with", "config", "setup", "guide"}
        for tag in potential_tags:
            if 3 <= len(tag) <= 20 and tag.lower() not in common_words:
                dynamic_tech_tags.add(tag)

    # 合并基础规则和动态规则
    merged_device_rules = {}
    for vendor in set(BASE_DEVICE_MODEL_RULES.keys()) | set(dynamic_device_models.keys()):
        base_models = BASE_DEVICE_MODEL_RULES.get(vendor, [])
        dynamic_models = list(dynamic_device_models.get(vendor, set()))
        # 去重并保留基础规则在前
        merged_models = list(dict.fromkeys(base_models + dynamic_models))
        merged_device_rules[vendor] = merged_models

    # 合并技术标签规则
    merged_tag_rules = BASE_TECH_TAG_RULES.copy()
    # 为动态标签创建简单映射（可根据需求自定义）
    for tag in dynamic_tech_tags:
        if tag.lower() not in merged_tag_rules:
            merged_tag_rules[tag.lower()] = f"自动识别-{tag}"

    logger.info(
        f"动态规则合并完成：设备型号{sum(len(v) for v in merged_device_rules.values())}个，技术标签{len(merged_tag_rules)}个")
    return merged_device_rules, merged_tag_rules


# -------------------------- 核心函数 --------------------------
def load_multiclass_docs(data_dir: str = r"app/data/init_docs") -> list:
    ocr_processor = GPT4VOCRProcessor()
    all_docs = []

    # 打印实际遍历的目录，确认路径正确
    data_path = Path(data_dir)
    logger.info(f"开始加载文档，根目录：{data_path.resolve()}")  # 打印绝对路径
    if not data_path.exists():
        logger.error(f"根目录不存在：{data_path.resolve()}")
        return all_docs

    # 遍历根目录下的所有子目录（修复：确保只遍历一级子目录）
    for item in data_path.iterdir():
        if not item.is_dir():
            logger.debug(f"跳过非目录文件：{item.name}")
            continue

        subdir_name = item.name
        logger.info(f"发现子目录：{subdir_name}（路径：{item.resolve()}）")  # 打印子目录名和路径

        # 检查子目录是否在CATEGORY_MAP中
        if subdir_name not in CATEGORY_MAP:
            logger.warning(f"未知子目录{subdir_name}，跳过（需在CATEGORY_MAP中定义）")
            continue

        # 加载该子目录下的文档
        category = CATEGORY_MAP[subdir_name]
        logger.info(f"开始加载分类[{category}]：{item.resolve()}")

        try:
            reader = SimpleDirectoryReader(
                input_dir=item.resolve(),  # 使用绝对路径
                required_exts=[".json", ".md", ".pdf", ".txt"],
                recursive=True  # 若子目录下还有子目录，需开启recursive
            )
            dir_docs = reader.load_data()
        except Exception as e:
            logger.error(f"加载子目录{subdir_name}失败：{str(e)}", exc_info=True)
            continue

        if not dir_docs:
            logger.warning(f"分类[{category}]目录为空，跳过：{item.resolve()}")
            continue

        # 为每个文档添加分类元数据（关键：确保分类被写入）
        for doc in dir_docs:
            doc.metadata["category"] = category  # 强制写入分类
            doc.metadata["subdir"] = subdir_name
            doc.metadata["root_dir"] = data_path.resolve()  # 新增：记录根目录，便于调试
            all_docs.append(doc)

        logger.info(f"分类[{category}]加载完成，共{len(dir_docs)}份文档")

    logger.info(f"所有分类文档加载完成，总计{len(all_docs)}份原始文档")
    return all_docs


def split_into_nodes(docs: list) -> list:
    """分割文档为Node节点（优化技术文档分割逻辑）"""
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
        # 针对命令行/配置步骤优化分割符（优先按空行/分号分割）
        # separators=["\n\n", "\n", "；", "。", " ", ""]
    )
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"文档分割完成，生成{len(nodes)}个Node节点")
    return nodes


def add_metadata_to_nodes(nodes: list, device_rules: dict, tag_rules: dict) -> list:
    """增强元数据：补充分类、厂商、设备型号、技术标签（使用动态规则）"""
    for node in nodes:
        # 1. 从文档元数据继承分类（核心：保留分类信息）
        category = node.metadata.get("category", "unknown")
        subdir = node.metadata.get("subdir", "unknown")

        # 2. 识别厂商（华为/思科）
        vendor = "huawei" if category.startswith("huawei") else "cisco"

        # 3. 提取设备型号（使用合并后的规则）
        doc_name = node.metadata.get("file_name", "").lower()
        node_text = node.text.lower()
        device_model = "unknown"

        # 优先匹配基础规则，再匹配动态规则
        for model in device_rules[vendor]:
            if model.lower() in doc_name or model.lower() in node_text:
                device_model = model
                break

        # 4. 提取技术标签（使用合并后的规则）
        tech_tag = "unknown"
        for keyword, tag in tag_rules.items():
            if keyword in doc_name or keyword in node_text:
                tech_tag = tag
                break

        # 5. 更新元数据（包含分类信息，供检索时使用）
        node.metadata.update({
            "category": category,  # 分类标签（如huawei_solutions）
            "vendor": vendor,  # 厂商（huawei/cisco）
            "tech_tag": tech_tag,  # 技术标签
            "device_model": device_model,  # 设备型号
            "is_public": True
        })

    logger.info("Node节点元数据添加完成（含分类信息）")
    return nodes


def generate_vectors(nodes: list) -> list:
    """为Node节点生成向量（保持原有逻辑，适配批量处理）"""
    embedding_model = SentenceTransformer("BAAI/bge-large-zh")
    batch_size = 64
    all_vectors = []

    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i + batch_size]
        texts = [node.text for node in batch_nodes]
        vectors = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        all_vectors.extend(vectors)
        logger.info(f"已生成{i + len(batch_nodes)}/{len(nodes)}个向量")

    return all_vectors


def init_kb():
    """初始化多分类知识库（支持动态扩展规则）"""
    try:
        # 1. 加载多分类文档（按子目录识别分类）
        docs = load_multiclass_docs()
        if not docs:
            logger.warning("未找到任何文档，初始化终止")
            return

        # 2. 动态收集并合并规则
        device_rules, tag_rules = collect_dynamic_rules(docs)

        # 3. 分割为Node节点
        nodes = split_into_nodes(docs)

        # 4. 添加增强元数据（使用动态规则）
        nodes = add_metadata_to_nodes(nodes, device_rules, tag_rules)

        # 5. 生成向量
        vectors = generate_vectors(nodes)

        # 6. 构造插入数据（包含分类信息）
        insert_docs = [
            {
                "text": nodes[i].text,
                "vector": vectors[i],
                "category": nodes[i].metadata["category"],
                "vendor": nodes[i].metadata["vendor"],
                "tech_tag": nodes[i].metadata["tech_tag"],
                "device_model": nodes[i].metadata["device_model"],
                "source": nodes[i].metadata.get("file_name", "unknown"),
                "source_url": nodes[i].metadata.get("source_url", "unknown"),
                "is_public": nodes[i].metadata["is_public"],
                "title": nodes[i].metadata.get("title", "未知标题")
            } for i in range(len(nodes))
        ]

        # 7. 插入Faiss向量库
        FAISS_DIR = r"app/data/faiss_index"
        faiss_client = FaissClient(index_path=FAISS_DIR)
        inserted_count = faiss_client.insert_documents(insert_docs)

        # 8. 优化索引
        index_manager = FaissIndexManager(faiss_client)
        index_manager.optimize()

        logger.info(f"多分类知识库初始化完成，共插入{inserted_count}条文档（含分类信息）")

        try:
            from app.modules.retrieval.build_bm25_from_faiss import main as build_bm25_main
            build_bm25_main()  # 传递已构造的文档数据
            logger.info("BM25索引自动构建完成")
        except Exception as e:
            logger.error(f"BM25构建失败：{str(e)}", exc_info=True)

    except Exception as e:
        logger.error(f"初始化失败：{str(e)}", exc_info=True)


if __name__ == "__main__":
    init_kb()  # 入口改为多分类初始化
