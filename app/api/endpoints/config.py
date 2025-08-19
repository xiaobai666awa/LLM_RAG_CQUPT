
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services.chat_logger import ChatLogger

# （可从配置文件或数据库加载）
SUPPORTED_MODELS = [
    {"type": "openai", "name": "gpt-4-turbo"},
    {"type": "openai", "name": "gpt-4o"},
    {"type": "anthropic", "name": "claude-3-opus"},
    {"type": "anthropic", "name": "claude-3-opus"},
    {"type": "siliconflow","name":"Qwen/Qwen3-8B"}

]
conversation_configs =ChatLogger()
router = APIRouter(prefix="/config", tags=["config"])

@router.get("/models")
def list_supported_models() -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            " conversation models": SUPPORTED_MODELS,
        }
    )

@router.get("/{conversation_id}")
def get_config(conversation_id: str) -> JSONResponse:
    model= conversation_configs.get_model(conversation_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"No configuration found for conversation `{conversation_id}`")

    return JSONResponse(
        status_code=200,
        content={
            "conversation_id": conversation_id,
            "model": model
        }
    )


@router.post("/{conversation_id}/update")
def update_config(conversation_id: str, payload: Dict[str, Any]) -> JSONResponse:
    if "model" not in payload:
        raise HTTPException(status_code=400, detail="`model` field is required in payload.")

    conversation_configs.update_model(conversation_id, payload["model"])
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Configuration updated for conversation {conversation_id}."
        }
    )

