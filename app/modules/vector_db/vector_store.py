from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Lock
import os
import json
import faiss
import numpy as np
from pathlib import Path

# 向量维度（根据BGE-large-zh模型输出调整）
VECTOR_DIM = 1024
# 元数据存储路径（与Faiss索引同目录）
METADATA_FILE = "metadata.json"


class VectorStore:
    """基于Faiss的向量存储管理类，支持持久化存储和向量检索"""

    def __init__(self, index_path: str = "../data/faiss_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Faiss索引文件路径
        self.index_file = self.index_path / "index.faiss"

        # 元数据存储：记录向量与文档的映射关系
        # 结构: {
        #   "upload_id_to_ids": {upload_id: [vector_id1, vector_id2, ...]},
        #   "vector_metadata": {vector_id: {"chunk": str, "metadata": dict, ...}}
        # }
        self.metadata_path = self.index_path / METADATA_FILE
        self._load_metadata()

        # 加载或创建Faiss索引
        self._load_or_create_index()

        # 线程锁确保并发安全
        self._lock = Lock()
        # 向量ID计数器（确保唯一）
        self._next_id = max(self.vector_metadata.keys(), default=-1) + 1

    def _load_or_create_index(self):
        """加载已有索引或创建新索引"""
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            print(f"已加载Faiss索引，包含{self.index.ntotal}个向量")
        else:
            # 创建L2距离的Flat索引（适合中小规模数据）
            self.index = faiss.IndexFlatL2(VECTOR_DIM)
            print(f"创建新的Faiss索引，维度: {VECTOR_DIM}")

    def _load_metadata(self):
        """加载元数据（向量与文档的映射关系）"""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.upload_id_to_ids = data.get("upload_id_to_ids", {})
                self.vector_metadata = {int(k): v for k, v in data.get("vector_metadata", {}).items()}
        else:
            self.upload_id_to_ids = {}
            self.vector_metadata = {}

    def _save_metadata(self):
        """保存元数据到文件"""
        data = {
            "upload_id_to_ids": self.upload_id_to_ids,
            "vector_metadata": {str(k): v for k, v in self.vector_metadata.items()}
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save_index(self):
        """保存Faiss索引到文件"""
        faiss.write_index(self.index, str(self.index_file))

    def add(self, upload_id: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any]) -> None:
        """添加文档向量到Faiss索引"""
        with self._lock:
            if upload_id in self.upload_id_to_ids:
                raise ValueError(f"upload_id '{upload_id}' 已存在，如需更新请使用update方法")

            # 验证输入合法性
            if len(chunks) != len(embeddings):
                raise ValueError("chunks与embeddings长度不匹配")

            # 转换向量为numpy数组（Faiss要求格式）
            vectors = np.array(embeddings, dtype=np.float32)
            if vectors.shape[1] != VECTOR_DIM:
                raise ValueError(f"向量维度必须为{VECTOR_DIM}，实际为{vectors.shape[1]}")

            # 分配向量ID
            vector_ids = list(range(self._next_id, self._next_id + len(vectors)))
            self._next_id += len(vectors)

            # 1. 添加向量到Faiss索引
            self.index.add(vectors)

            # 2. 记录元数据映射
            self.upload_id_to_ids[upload_id] = vector_ids
            for idx, vec_id in enumerate(vector_ids):
                self.vector_metadata[vec_id] = {
                    "chunk": chunks[idx],
                    "metadata": metadata,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }

            # 3. 持久化存储
            self._save_index()
            self._save_metadata()

    def update(self, upload_id: str, chunks: List[str], embeddings: List[List[float]],
               metadata: Dict[str, Any]) -> None:
        """更新文档向量（先删除旧向量再添加新向量）"""
        with self._lock:
            if upload_id not in self.upload_id_to_ids:
                raise KeyError(f"upload_id '{upload_id}' 不存在，无法更新")

            # 1. 删除旧向量
            old_ids = self.upload_id_to_ids[upload_id]
            self._delete_vectors(old_ids)

            # 2. 添加新向量（复用add逻辑）
            self.add(upload_id, chunks, embeddings, metadata)

            # 3. 更新时间戳
            for vec_id in self.upload_id_to_ids[upload_id]:
                self.vector_metadata[vec_id]["updated_at"] = datetime.now().isoformat()

            self._save_metadata()

    def _delete_vectors(self, vector_ids: List[int]) -> None:
        """从索引中删除指定ID的向量（Faiss需要重建索引实现删除）"""
        # 1. 收集剩余向量和元数据
        remaining_vectors = []
        remaining_metadata = {}
        remaining_upload_id_map = {}

        # 遍历所有向量ID，排除要删除的ID
        all_ids = list(self.vector_metadata.keys())
        for vec_id in all_ids:
            if vec_id not in vector_ids:
                # 找到向量在索引中的位置（通过ID顺序）
                pos = all_ids.index(vec_id)
                remaining_vectors.append(self.index.reconstruct(pos))
                remaining_metadata[vec_id] = self.vector_metadata[vec_id]

        # 2. 重建索引
        self.index = faiss.IndexFlatL2(VECTOR_DIM)
        self.index.add(np.array(remaining_vectors, dtype=np.float32))

        # 3. 更新元数据映射
        for uid, ids in self.upload_id_to_ids.items():
            remaining_ids = [id for id in ids if id not in vector_ids]
            if remaining_ids:
                remaining_upload_id_map[uid] = remaining_ids

        self.upload_id_to_ids = remaining_upload_id_map
        self.vector_metadata = remaining_metadata

        # 4. 保存更改
        self._save_index()
        self._save_metadata()

    def delete(self, upload_id: str) -> bool:
        """删除指定upload_id的所有向量"""
        with self._lock:
            if upload_id not in self.upload_id_to_ids:
                return False

            # 删除关联的向量
            vector_ids = self.upload_id_to_ids[upload_id]
            self._delete_vectors(vector_ids)
            return True

    def list_documents(self) -> List[Dict[str, Any]]:
        """列出所有文档的元数据信息"""
        with self._lock:
            document_list = []
            for upload_id, vec_ids in self.upload_id_to_ids.items():
                if vec_ids:  # 确保有向量
                    # 取第一个向量的元数据作为文档代表
                    first_meta = self.vector_metadata[vec_ids[0]]["metadata"]
                    document_list.append({
                        "upload_id": upload_id,
                        "metadata": first_meta,
                        "chunk_count": len(vec_ids),
                        "created_at": self.vector_metadata[vec_ids[0]]["created_at"],
                        "updated_at": self.vector_metadata[vec_ids[0]]["updated_at"]
                    })
            # 按创建时间排序
            return sorted(document_list, key=lambda x: x["created_at"], reverse=True)

    def get(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """获取指定文档的完整信息（文本片段+向量+元数据）"""
        with self._lock:
            if upload_id not in self.upload_id_to_ids:
                return None

            vec_ids = self.upload_id_to_ids[upload_id]
            return {
                "chunks": [self.vector_metadata[vid]["chunk"] for vid in vec_ids],
                "embeddings": [self.index.reconstruct(self._get_vector_pos(vid)) for vid in vec_ids],
                "metadata": self.vector_metadata[vec_ids[0]]["metadata"],
                "created_at": self.vector_metadata[vec_ids[0]]["created_at"],
                "updated_at": self.vector_metadata[vec_ids[0]]["updated_at"]
            }

    def _get_vector_pos(self, vector_id: int) -> int:
        """获取向量在Faiss索引中的位置（ID到索引的映射）"""
        all_ids = sorted(self.vector_metadata.keys())
        return all_ids.index(vector_id)



    def _get_upload_id_by_vec_id(self, vec_id: int) -> Optional[str]:
        """通过向量ID反向查找对应的upload_id"""
        for upload_id, ids in self.upload_id_to_ids.items():
            if vec_id in ids:
                return upload_id
        return None
