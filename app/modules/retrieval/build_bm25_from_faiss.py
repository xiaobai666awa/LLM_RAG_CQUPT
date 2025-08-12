"""
把 Faiss 向量库的元数据（metadata.pkl）转换成 BM25 索引，方便后续进行 混合检索（向量 + BM25）
"""
import os
import pickle
import shutil
from app.modules.retrieval.bm25_index import BM25IndexManager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FAISS_DIR = ROOT_DIR / "data" / "faiss_index"
BM25_DIR = ROOT_DIR / "data" / "bm25_index"

def main():
    meta_path = os.path.join(FAISS_DIR, "metadata.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path}，请建Faiss库")

    # 读取 metadata
    meta = pickle.load(open(meta_path, "rb"))

    docs = []
    for i, m in enumerate(meta):
        docs.append({
            "doc_id": str(i),
            "text": m.get("text", ""),
            "tech_tag": m.get("tech_tag", ""),
            "device_model": m.get("device_model", ""),
            "is_public": bool(m.get("is_public", True)),
        })

    # 自动删除旧 BM25 索引
    if os.path.exists(BM25_DIR):
        print(f"[CLEAN] 删除旧 BM25 索引目录: {BM25_DIR}")
        shutil.rmtree(BM25_DIR)

    # 创建并写入 BM25 索引
    bm25 = BM25IndexManager(index_dir=BM25_DIR)
    bm25.add_documents(docs)
    print(f"[DONE] BM25 写入完成: 共 {len(docs)} 条")

if __name__ == "__main__":
    main()
