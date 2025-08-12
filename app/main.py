import fastapi
from fastapi import APIRouter

from api.endpoints import chat
from api.endpoints import config
from api.endpoints import vector
def init_app():
    return "应用初始化完毕"

def register_routes(app: fastapi.FastAPI):
    app.include_router(chat.router)
    app.include_router(config.router)
    app.include_router(vector.router)
    return "路由注册完毕"

app=fastapi.FastAPI()
print("正在注册路由")
print(register_routes(app))

# from flask import Flask, request, jsonify
# from app.modules.retrieval import HybridRetriever
#
# app = Flask(__name__)
#
# # 初始化混合检索器（可调节权重）
# retriever = HybridRetriever(vector_weight=0.7, bm25_weight=0.3)

# @app.route("/search", methods=["POST"])
# def search():
#     try:
#         data = request.get_json()
#         query = data.get("query", "")
#         filters = data.get("filters", None)  # 如 {"device_model": "S5720", "is_public": True}
#         top_k = data.get("top_k", 5)
#
#         if not query:
#             return jsonify({"error": "缺少 query 参数"}), 400
#
#         results = retriever.retrieve(query, user_filters=filters, top_k=top_k)
#         return jsonify({"results": results})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == "__main__":
#     app.run(host="192.168.0.192", port=8080, debug=True)
#
#
# '''
# 接口传参：
# {
#   "query": "API网关",
#   "filters": {"device_model": "unknown", "is_public": true},
#   "top_k": 5
# }
# '''
