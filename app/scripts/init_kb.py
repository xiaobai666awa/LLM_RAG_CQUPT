import json
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

from app.core.model import get_embedding_model
from app.modules.vector_db.client import FaissClient
from app.modules.vector_db.index_manager import FaissIndexManager
from app.common.logger import logger
from app.common.utils import GPT4VOCRProcessor   # 引入OCR工具（处理图片/PDF）

# -------------------------- 核心配置 --------------------------
# 1. 分类映射：子目录名 → 分类标签（需与DATA目录结构对应）
CATEGORY_MAP = {
    "knowledge": "huawei_knowledge",  # 华为-基础知识
    "RFC": "huawei_rfc",  # 华为-RFC文档
    "solutions": "huawei_solutions",  # 华为-配置方案
    "cisco": "cisco_config"  # 思科-配置手册
}

# 2. 设备型号提取规则（分厂商）
DEVICE_MODEL_RULES = {
    "huawei": ["S5720", "S12700", "NE40E", "AR650"],  # 华为设备型号
    "cisco": ["Catalyst 9300", "ISR 4000", "Nexus 9000"]  # 思科设备型号
}

# 3. 技术标签提取规则（扩展多厂商）
TECH_TAG_RULES = {
    "vlan": "交换机配置-VLAN",
    "ospf": "路由协议-OSPF",
    "bgp": "路由协议-BGP",
    "acl": "安全策略-ACL",
    "nat": "地址转换-NAT"
}


# -------------------------- 核心函数 --------------------------
def load_multiclass_docs(data_dir: str = r"app/data/init_docs") -> list:
    """
    加载多分类文档（支持子目录自动识别分类）
    处理逻辑：遍历子目录 → 按目录名映射分类 → 提取文本（含OCR处理）
    """
    ocr_processor = GPT4VOCRProcessor()
    all_docs = []

    for subdir in Path(data_dir).iterdir():
        if not subdir.is_dir():
            continue  # 跳过文件

        subdir_name = subdir.name
        if subdir_name not in CATEGORY_MAP:
            logger.warning(f"未知子目录{subdir_name}，跳过")
            continue

        category = CATEGORY_MAP[subdir_name]
        logger.info(f"加载分类[{category}]：{subdir}")

        # 加载目录下的文档
        reader = SimpleDirectoryReader(
            input_dir=subdir,
            required_exts=[".json", ".md", ".pdf"],
            recursive=True
        )
        dir_docs = reader.load_data()

        # ✨ 新增：跳过空目录
        if not dir_docs:
            logger.warning(f"分类[{category}]目录为空，跳过：{subdir}")
            continue  # 无文档时跳过，不中断流程

        # 处理文档：对PDF进行OCR（防止扫描件无文本）
        for doc in dir_docs:
            doc.metadata["category"] = category
            doc.metadata["subdir"] = subdir_name

            if doc.metadata.get("file_extension") == ".json":
                try:
                    json_content = json.loads(doc.text)
                    doc.metadata["source_url"] = json_content.get("url")  # 提取URL
                    doc.metadata["title"] = json_content.get("title")  # 提取文档标题
                except Exception as e:
                    logger.error(f"解析JSON失败：{doc.metadata['file_name']}，错误：{e}")

            # 处理PDF：使用GPT-4V提取文字（关键修改）
            if doc.metadata.get("file_extension") == ".pdf":
                try:
                    # 调用GPT-4V OCR
                    ocr_text = ocr_processor.extract_text(doc.metadata["file_path"])
                    doc.text = ocr_text.strip()
                    logger.debug(f"GPT-4V OCR处理完成：{doc.metadata['file_name']}")
                except Exception as e:
                    logger.error(f"GPT-4V OCR失败{doc.metadata['file_name']}：{str(e)}")

        all_docs.extend(dir_docs)
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


def add_metadata_to_nodes(nodes: list) -> list:
    """增强元数据：补充分类、厂商、设备型号、技术标签"""
    for node in nodes:
        # 1. 从文档元数据继承分类（核心：保留分类信息）
        category = node.metadata.get("category", "unknown")
        subdir = node.metadata.get("subdir", "unknown")

        # 2. 识别厂商（华为/思科）
        vendor = "huawei" if category.startswith("huawei") else "cisco"

        # 3. 提取设备型号（分厂商匹配）
        doc_name = node.metadata.get("file_name", "")
        device_model = "unknown"
        for model in DEVICE_MODEL_RULES[vendor]:
            if model.lower() in doc_name.lower():
                device_model = model
                break

        # 4. 提取技术标签（支持多关键词）
        tech_tag = "unknown"
        for keyword, tag in TECH_TAG_RULES.items():
            if keyword in doc_name.lower() or keyword in node.text.lower():
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
    """初始化多分类知识库（替代原init_huawei_kb）"""
    try:
        # 1. 加载多分类文档（按子目录识别分类）
        docs = load_multiclass_docs()
        if not docs:
            logger.warning("未找到任何文档，初始化终止")
            return

        # 2. 分割为Node节点
        nodes = split_into_nodes(docs)

        # 3. 添加增强元数据（含分类）
        nodes = add_metadata_to_nodes(nodes)

        # 4. 生成向量
        vectors = generate_vectors(nodes)

        # 5. 构造插入数据（包含分类信息）
        insert_docs = [
            {
                "text": nodes[i].text,
                "vector": vectors[i],
                "category": nodes[i].metadata["category"],  # 核心：存储分类
                "vendor": nodes[i].metadata["vendor"],
                "tech_tag": nodes[i].metadata["tech_tag"],
                "device_model": nodes[i].metadata["device_model"],
                "source": nodes[i].metadata.get("file_name", "unknown"),
                "source_url": nodes[i].metadata.get("source_url", "unknown"),
                "is_public": nodes[i].metadata["is_public"]
            } for i in range(len(nodes))
        ]

        # 6. 插入Faiss向量库
        FAISS_DIR = r"app/data/faiss_index"
        faiss_client = FaissClient(index_path=FAISS_DIR)
        inserted_count = faiss_client.insert_documents(insert_docs)

        # 7. 优化索引
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
