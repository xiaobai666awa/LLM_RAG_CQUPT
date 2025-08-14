import json
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

from app.modules.vector_db.client import FaissClient
from app.modules.vector_db.index_manager import FaissIndexManager
from app.common.logger import logger

def load_huawei_docs(data_dir: str = r"../data/init_docs") -> list:
    ## 读取数据，设置绝对路径


    """加载华为技术文档（支持JSON/Markdown/PDF）"""
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".json", ".md", ".pdf"],
        recursive=True  # 递归加载子目录
    )
    docs = reader.load_data()
    logger.info(f"从{data_dir}加载了{len(docs)}份原始文档")
    return docs

def split_into_nodes(docs: list) -> list:
    """将文档分割为Node节点（保留技术术语完整性）"""
    splitter = SentenceSplitter(
        chunk_size=512,  # 每个节点约512字符（适合技术文档）
        chunk_overlap=50  # 重叠50字符，避免分割技术步骤
        # separators=["\n\n", "\n", "。", "；", " ", ""]  # 中文分割符
        ## 默认分隔符规则，默认分隔符针对多种语言优化，现在默认算法，切分更灵活，保留上下文

    )
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"文档分割完成，生成{len(nodes)}个Node节点")
    return nodes

def add_metadata_to_nodes(nodes: list) -> list:
    """为Node节点添加元数据（tech_tag/device_model等）"""
    # 示例：从文档名称提取设备型号（实际可根据内容用NLP提取）
    for node in nodes:
        doc_name = node.metadata.get("file_name", "")
        # 提取设备型号（如"S5720配置手册.json" → "S5720"）
        device_model = "unknown"
        for model in ["S5720", "S12700", "NE40E"]:  # 华为常见设备型号
            if model in doc_name:
                device_model = model
                break
        # 提取技术标签（如"VLAN配置"）
        tech_tag = "unknown"
        if "vlan" in doc_name.lower():
            tech_tag = "交换机配置-VLAN"
        elif "ospf" in doc_name.lower():
            tech_tag = "路由协议-OSPF"
        
        # 添加元数据
        node.metadata.update({
            "tech_tag": tech_tag,
            "device_model": device_model,
            "is_public": True
        })
    logger.info("Node节点元数据添加完成")
    return nodes

def generate_vectors(nodes: list) -> list:
    """为Node节点生成向量（BGE-large-zh模型）"""

    embedding_model = SentenceTransformer(r"D:\py_models\em_model", device="cuda")
    # 调用本地模型


    # 批量生成向量（每次batch_size条，避免内存溢出）
    batch_size = 64
    all_vectors = []

    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i:i+batch_size]
        texts = [node.text for node in batch_nodes]
        # vectors = embedding_model.embed_documents(texts)
        vectors = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        all_vectors.extend(vectors)
        logger.info(f"已生成{i+len(batch_nodes)}/{len(nodes)}个向量")
    return all_vectors

def init_huawei_kb():
    """初始化华为技术文档知识库"""
    try:
        # 1. 加载原始文档
        docs = load_huawei_docs()
        if not docs:
            logger.warning("未找到文档，初始化终止")
            return
        
        # 2. 分割为Node节点
        nodes = split_into_nodes(docs)
        
        # 3. 添加元数据
        nodes = add_metadata_to_nodes(nodes)
        
        # 4. 生成向量
        vectors = generate_vectors(nodes)
        
        # 5. 构造插入数据
        insert_docs = [
            {
                "text": nodes[i].text,
                "vector": vectors[i],
                "tech_tag": nodes[i].metadata["tech_tag"],
                "device_model": nodes[i].metadata["device_model"],
                "source": nodes[i].metadata.get("file_name", "unknown"),
                "is_public": nodes[i].metadata["is_public"]
            } for i in range(len(nodes))
        ]

        # 6. 插入向量数据库（Faiss）
        FAISS_DIR = r"D:\python object\LLAMAI_VECTOR\data\faiss_index"
        ## 设置一下向量库的路径

        faiss_client = FaissClient(index_path=FAISS_DIR)  # 实例化Faiss客户端
        inserted_count = faiss_client.insert_documents(insert_docs)

        # 7. 优化索引（Faiss无需单独创建索引，直接优化）
        index_manager = FaissIndexManager(faiss_client)
        index_manager.optimize()

        logger.info(f"知识库初始化完成，共插入{inserted_count}条文档")

    except Exception as e:
        logger.error(f"初始化失败：{str(e)}")

if __name__ == "__main__":
    init_huawei_kb()
