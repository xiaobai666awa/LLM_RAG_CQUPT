
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# （可替换为数据库）
conversation_configs: Dict[str, Dict[str, Any]] = {
    "001": {
        "model": {
            "provider": "qwen",
            "name": "qwen3-8b"
        }
    }
}


# （可从配置文件或数据库加载）
SUPPORTED_MODELS = [
    {"provider": "openai", "name": "gpt-4-turbo"},
    {"provider": "openai", "name": "gpt-4o"},
    {"provider": "anthropic", "name": "claude-3-opus"}
]

router = APIRouter()

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
    config = conversation_configs.get(conversation_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"No configuration found for conversation `{conversation_id}`")

    return JSONResponse(
        status_code=200,
        content={
            "conversation_id": conversation_id,
            "model": config.get("model")
        }
    )


@router.post("/{conversation_id}/update")
def update_config(conversation_id: str, payload: Dict[str, Any]) -> JSONResponse:
    if "model" not in payload:
        raise HTTPException(status_code=400, detail="`model` field is required in payload.")

    conversation_configs[conversation_id] = {
        **conversation_configs.get(conversation_id, {}),
        "model": payload["model"]
    }

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": f"Configuration updated for conversation {conversation_id}."
        }
    )



