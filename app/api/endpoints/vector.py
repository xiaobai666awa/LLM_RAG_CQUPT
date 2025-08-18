# app/api/endpoints/vector.py

"""
main.py
# 导入并挂载路由
from app.api.endpoints.vector import router as vector_router
app.include_router(vector_router, prefix="/vector")
"""

"""
# 1. add：upload_id 放在 query，file 用 -F 上传
curl -v \
  -X POST "http://localhost:8000/vector/add?upload_id=sample01" \
  -F "file=@sample.txt"

# 2. update：同理
curl -v \
  -X POST "http://localhost:8000/vector/update?upload_id=sample01" \
  -F "file=@sample.txt"

# 3. delete：upload_id 放在 query
curl -v \
  -X DELETE "http://localhost:8000/vector/delete?upload_id=sample01"

# 4. list：直接 GET
curl -v http://localhost:8000/vector/list
"""

from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import logging
from llama_index.core.readers import SimpleDirectoryReader  # 新增：用于读取单文件

# 导入初始化知识库的核心函数
from app.scripts.init_kb import (
    split_into_nodes,
    add_metadata_to_nodes,
    generate_vectors
)
from app.modules.vector_db.vector_store import VectorStore

router = APIRouter()
RAW_DOC_DIR = Path("./data/raw_documents")
vs = VectorStore()
logger = logging.getLogger(__name__)  # 初始化日志


def load_single_file(file_path: Path) -> list:
    """加载单个文件（复用初始化逻辑，支持JSON/Markdown/PDF）"""
    try:
        # 使用SimpleDirectoryReader读取单个文件
        reader = SimpleDirectoryReader(
            input_files=[str(file_path)],
            required_exts=[".json", ".md", ".pdf"]
        )
        docs = reader.load_data()
        logger.info(f"已加载文件: {file_path.name}, 文档数量: {len(docs)}")
        return docs
    except Exception as e:
        logger.error(f"文件加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件解析错误: {str(e)}")


def _embed_and_store(file_path: Path, upload_id: str, mode: str):
    """后台任务：使用初始化函数处理文档→切分→生成向量→存储"""
    try:
        # 1. 加载单个文件（替换原测试数据生成）
        docs = load_single_file(file_path)
        if not docs:
            logger.warning(f"文件内容为空: {file_path}")
            return

        # 2. 分割为Node节点（复用初始化的切分逻辑）
        nodes = split_into_nodes(docs)

        # 3. 为节点添加元数据（技术标签、设备型号等）
        nodes = add_metadata_to_nodes(nodes)

        # 4. 生成向量（复用BGE-large-zh模型）
        vectors = generate_vectors(nodes)

        # 5. 准备存储数据（整合文本、向量和元数据）
        chunks = [node.text for node in nodes]
        docs_meta = {
            "upload_id": upload_id,
            "source": file_path.name,
            # 从节点元数据中提取公共属性
            "tech_tag": nodes[0].metadata.get("tech_tag", "unknown") if nodes else "unknown",
            "device_model": nodes[0].metadata.get("device_model", "unknown") if nodes else "unknown"
        }

        # 6. 存储到向量数据库
        if mode == "add":
            vs.add(upload_id, chunks, vectors, docs_meta)
        elif mode == "update":
            vs.update(upload_id, chunks, vectors, docs_meta)

        logger.info(f"{mode}成功: upload_id={upload_id}, 处理节点数={len(nodes)}")

    except Exception as e:
        logger.error(f"后台处理失败: {str(e)}")


@router.post("/add")
def add_file_to_vector(
    file: UploadFile,
    upload_id: str,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 保存文件到本地
    dest_dir = RAW_DOC_DIR / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file.filename
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 异步处理文档并存储向量
    background_tasks.add_task(_embed_and_store, file_path, upload_id, mode="add")

    return JSONResponse(
        status_code=200,
        content={
            "message": "文件上传成功，正在处理向量...",
            "id": upload_id,
            "filename": file.filename
        }
    )


@router.post("/update")
def update_vector(
    file: UploadFile,
    upload_id: str,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    dest_dir = RAW_DOC_DIR / upload_id
    if not dest_dir.exists():
        raise HTTPException(status_code=404, detail=f"upload_id `{upload_id}` 不存在")

    # 删除旧文件，保存新文件
    shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_path = dest_dir / file.filename
    with file_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # 异步更新向量
    background_tasks.add_task(_embed_and_store, file_path, upload_id, mode="update")

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"upload_id `{upload_id}` 已更新，正在重新处理向量..."
        }
    )


@router.delete("/delete")
def delete_vector(upload_id: str) -> JSONResponse:
    # 从向量库删除
    deleted = vs.delete(upload_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"未找到 upload_id `{upload_id}`")

    # 删除本地文件
    dest_dir = RAW_DOC_DIR / upload_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    return JSONResponse(
        status_code=200,
        content={"message": f"upload_id `{upload_id}` 已完全删除"}
    )


@router.get("/list")
def list_vector_files() -> JSONResponse:
    # 获取所有文档信息（包含元数据）
    documents = vs.list_documents()
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "count": len(documents),
            "documents": documents
        }
    )



def _embed_and_store(file_path: Path, upload_id: str, mode: str):
    """
    后台任务：抽取 → 切分 → 嵌入 → 存到 VectorStore
    """
    try:
        # 1. 抽取文本（使用本地load_single_file）
        docs = load_single_file(file_path)
        if not docs:
            logger.warning(f"文件{file_path}提取不到文本内容")
            return

        # 2. 切分文本（得到TextNode对象列表）
        from app.scripts.init_kb import split_into_nodes
        nodes = split_into_nodes(docs)
        if not nodes:
            logger.warning(f"文件{file_path}分割后无有效内容")
            return

        # 3. 添加元数据
        from app.scripts.init_kb import add_metadata_to_nodes
        nodes = add_metadata_to_nodes(nodes)

        # 4. 生成向量（已修正TextNode访问方式）
        from app.scripts.init_kb import generate_vectors
        embeddings = generate_vectors(nodes)

        # 验证向量维度
        if embeddings and len(embeddings[0]) != 1024:
            raise ValueError(f"向量维度错误，预期1024，实际{len(embeddings[0])}")

        # 5. 准备存储数据（关键修正：TextNode用.text和.metadata属性）
        chunks = [node.text for node in nodes]  # 修正这里
        docs_meta = {
            "upload_id": upload_id,
            "source": file_path.name,
            # 修正元数据访问方式：TextNode的metadata是属性
            "tech_tag": nodes[0].metadata.get("tech_tag", "unknown") if nodes else "unknown",
            "device_model": nodes[0].metadata.get("device_model", "unknown") if nodes else "unknown"
        }

        # 6. 存储到向量数据库
        from app.modules.vector_db.vector_store import VectorStore
        vs = VectorStore()
        if mode == "add":
            vs.add(upload_id, chunks, embeddings, docs_meta)
        elif mode == "update":
            vs.update(upload_id, chunks, embeddings, docs_meta)

        logger.info(f"{mode}成功：upload_id={upload_id}，生成{len(chunks)}个片段")

    except Exception as e:
        logger.error(f"处理文件{file_path}失败：{str(e)}", exc_info=True)


