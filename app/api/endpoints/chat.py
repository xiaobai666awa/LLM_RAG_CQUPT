# app/api/endpoints/chat.py

"""
main.py

from app.api.endpoints.chat import router as chat_router
app.include_router(chat_router)

"""
"""
# 1) start
curl -s localhost:8000/chat/start | jq

# 2) send（把上一步返回的 chat_id 填进来）
CID=123456
curl -s -X POST "localhost:8000/chat/$CID/message" \
  -H 'content-type: application/json' \
  -d '{"message":"hello pricing","history_limit":10}' | jq

# 3) get conversation
curl -s "localhost:8000/chat/$CID" | jq

# 4) list
curl -s "localhost:8000/chat/list" | jq

# 5) delete
curl -s -X DELETE "localhost:8000/chat/$CID" | jq

# 6) 再次 get 应 404
curl -s -i "localhost:8000/chat/$CID"

"""
import random
from datetime import datetime, timezone
from threading import Lock
from typing import Set

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 直接依赖你的三大服务
from app.services.chat_logger import ChatLogger
from app.services.conversation_manager import ConversationManager
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

# ========= 请求体 =========
class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    history_limit: int = Field(10, ge=0, le=100, description="截取上下文尾部条数，0 表示不限")

# ========= 模块级实例 =========
_logger = ChatLogger()
_conv = ConversationManager(_logger)
_rag = RAGPipeline()  # 模型就绪后替换为真实实现

# ========= 随机 chat_id 生成（带去重保障）=========
# 为了可读性，这里生成 6 位十进制随机数作为字符串，如 "483920"
# 如果你更喜欢长一些，改成 randint(10**8, 10**9-1) 等即可。
_id_lock = Lock()
_existing_ids: Set[str] = set()

def _hydrate_existing_ids_once() -> None:
    """进程首次使用时，把已存在的会话ID灌进集合（适配持久化 Logger）"""
    if _existing_ids:
        return
    try:
        items = _logger.list_conversations()
        for it in items:
            cid = str(it.get("conversation_id"))
            if cid:
                _existing_ids.add(cid)
    except Exception:
        # 内存 Logger 没有历史也没关系
        pass

def _new_chat_id() -> str:
    """生成随机 chat_id，避免冲突；多次尝试后仍冲突则抛错。"""
    _hydrate_existing_ids_once()
    MAX_TRIES = 20
    with _id_lock:
        for _ in range(MAX_TRIES):
            cid = str(random.randint(100000, 999999))  # 6 位数字
            if cid not in _existing_ids:
                _existing_ids.add(cid)
                return cid
    # 极端情况下仍冲突
    raise RuntimeError("Failed to allocate unique chat_id after multiple tries")

def _exists(conversation_id: str) -> bool:
    """判断会话是否存在（优先看内存集合，兜底走 Logger）"""
    if conversation_id in _existing_ids:
        return True
    # 兜底：从 logger 列表确认（适配不同 Logger 实现）
    try:
        ids = {str(x["conversation_id"]) for x in _logger.list_conversations()}
        if conversation_id in ids:
            _existing_ids.add(conversation_id)
            return True
    except Exception:
        pass
    return False

# ========= Endpoints =========

@router.get("/chat/start")
def start_conversation() -> JSONResponse:
    cid = _new_chat_id()
    _logger.create_conversation(cid)
    payload = {
        "message": "Chat session started successfully.",
        "chat_id": cid,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return JSONResponse(payload, status_code=status.HTTP_201_CREATED)

@router.get("/chat/list")
def list_conversations() -> JSONResponse:
    items = _logger.list_conversations()
    return JSONResponse({"items": items, "total": len(items)})

@router.post("/chat/{conversation_id}/message")
def send_message(conversation_id: str, msg: ChatMessageRequest) -> JSONResponse:
    # 严格要求：必须先 /chat/start
    if not _exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found. Call /chat/start first.")

    # 1) 记录用户消息
    _logger.append(conversation_id, {"role": "user", "content": msg.message})
    # 2) 获取上下文
    ctx = _conv.get_context(conversation_id, history_limit=msg.history_limit)
    # 3) RAG 生成
    answer = _rag.run(msg.message, ctx)
    # 4) 记录助手消息
    _logger.append(conversation_id, {"role": "assistant", "content": answer})

    return JSONResponse({"conversation_id": conversation_id, "answer": answer})

@router.get("/chat/{conversation_id}")
def get_conversation(conversation_id: str) -> JSONResponse:
    if not _exists(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    history = _logger.load(conversation_id)
    return JSONResponse({"conversation_id": conversation_id, "history": history})

@router.delete("/chat/{conversation_id}")
def delete_conversation(conversation_id: str) -> JSONResponse:
    # 即使 _exists 里没命中，也委托底层删除返回布尔值
    ok = _logger.delete(conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # 同步内存集合
    with _id_lock:
        _existing_ids.discard(conversation_id)
    return JSONResponse({"message": "Conversation deleted.", "conversation_id": conversation_id})

