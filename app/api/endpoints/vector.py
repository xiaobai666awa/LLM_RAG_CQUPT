# # app/api/endpoints/vector.py
#
# """
# main.py
# # 导入并挂载路由
# from app.api.endpoints.vector import router as vector_router
# app.include_router(vector_router, prefix="/vector")
# """
#
# """
# # 1. add：upload_id 放在 query，file 用 -F 上传
# curl -v \
#   -X POST "http://localhost:8000/vector/add?upload_id=sample01" \
#   -F "file=@sample.txt"
#
# # 2. update：同理
# curl -v \
#   -X POST "http://localhost:8000/vector/update?upload_id=sample01" \
#   -F "file=@sample.txt"
#
# # 3. delete：upload_id 放在 query
# curl -v \
#   -X DELETE "http://localhost:8000/vector/delete?upload_id=sample01"
#
# # 4. list：直接 GET
# curl -v http://localhost:8000/vector/list
# """
#
# from typing import Optional,List
# from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse
# from pathlib import Path
# import shutil
# import logging
#
# from app.core.utils import extract_text
# from app.embedding.splitter import split_text
# from app.embedding.embedder import Embedder
# from app.embedding.vector_store import VectorStore
#
# router = APIRouter()
# RAW_DOC_DIR = Path("app/data/raw_documents")
# vs = VectorStore()
#
# @router.post("/add")
# def add_file_to_vector(
#     file: UploadFile ,
#     upload_id: str,
#     background_tasks: BackgroundTasks
# ) -> JSONResponse:
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="文件名不能为空")
#
#     # 保存文件
#     dest_dir = RAW_DOC_DIR / upload_id
#     dest_dir.mkdir(parents=True, exist_ok=True)
#     file_path = dest_dir / file.filename
#     with file_path.open("wb") as f:
#         shutil.copyfileobj(file.file, f)
#
#     # 异步向量化并 add
#
#     background_tasks.add_task(_embed_and_store, file_path, upload_id, mode="add")
#
#     return JSONResponse(
#         status_code=200,
#         content={
#             "message":  "File uploaded successfully.",
#             "id":       upload_id,
#             "filename": file.filename,
#             "filepath": str(file_path),
#         },
#     )
#
#
# @router.post("/update")
# def update_vector(
#     file: UploadFile,
#     upload_id: str,
#     background_tasks: BackgroundTasks
# ) -> JSONResponse:
#
#     dest_dir = RAW_DOC_DIR / upload_id
#     if not dest_dir.exists():
#         raise HTTPException(status_code=404, detail=f"upload_id `{upload_id}` 不存在")
#
#     # 删除旧文件
#     shutil.rmtree(dest_dir)
#     dest_dir.mkdir(parents=True, exist_ok=True)
#
#     # 保存新文件
#     file_path = dest_dir / file.filename
#     with file_path.open("wb") as out:
#         shutil.copyfileobj(file.file, out)
#
#     # 异步向量化并 update
#     background_tasks.add_task(_embed_and_store, file_path, upload_id, mode="update")
#
#     return JSONResponse(
#         status_code=200,
#         content={
#             "status":  "success",
#             "message": f"upload_id `{upload_id}` 的文件已更新并开始向量化"
#         },
#     )
#
#
# @router.delete("/delete")
# def delete_vector(
#     upload_id: str) -> JSONResponse:
#     deleted = vs.delete(upload_id)
#     if not deleted:
#         raise HTTPException(status_code=404, detail=f"No vectors found for upload_id `{upload_id}`")
#
#     # 删除磁盘文件夹
#     dest_dir = RAW_DOC_DIR / upload_id
#     if dest_dir.exists():
#         shutil.rmtree(dest_dir)
#
#     return JSONResponse(
#         status_code=200,
#         content={
#             "status":  "success",
#             "message": f"Document with upload_id={upload_id} has been deleted."
#         }
#     )
#
#
# @router.get("/list")
# def list_vector_files() -> JSONResponse:
#     # 获取所有文档信息
#     documents = vs.list_documents()
#
#     # 返回文档信息的 JSON 格式响应
#     return JSONResponse(
#         status_code=200,
#         content={
#             "status": "success",
#             "documents": documents
#         }
#     )
#
#
# def _embed_and_store(file_path: Path, upload_id: str, mode: str):
#     """
#     后台任务：抽取 → 切分 → 嵌入 → 存到 VectorStore
#     """
#
#     """
#     # 1. 抽文本
#     text = extract_text(file_path)
#     # 2. 切分
#     chunks = split_text(text)
#
#     # 3. 嵌入
#     emb = Embedder()
#     emb.load_model(provider="fake", model_name="fake")
#     embeddings = emb.batch_embed(chunks)
#     """
#
#
# #### 测试假数据（调试用）
#     chunks = ['a','b']
#     embeddings = [[1,2,3],[4,5,6]]
#
#
#     # 4. 准备 metadata
#     docs_meta = {"upload_id": upload_id, "source": file_path.name}
#
#
#     if mode == "add":
#         vs.add(upload_id, chunks, embeddings, docs_meta)
#     elif mode == "update":
#         vs.update(upload_id, chunks, embeddings, docs_meta)
#
#
