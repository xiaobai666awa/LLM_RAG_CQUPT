import fastapi
from fastapi import APIRouter

from api import Chat


def register_routes(app: fastapi.FastAPI):
    app.include_router(Chat.router)
    return "路由注册完毕"

app=fastapi.FastAPI()
print("正在注册路由")
print(register_routes(app))

