from fastapi import status
from pydantic import BaseModel

class AppException(BaseException):
    """应用基类异常"""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class RetrievalException(AppException):
    """检索相关异常（如向量库连接失败、无结果）"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)

class LLMException(AppException):
    """大模型调用异常（如超时、API错误）"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_504_GATEWAY_TIMEOUT)

class ValidationException(AppException):
    """参数验证异常（如输入格式错误）"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class PermissionException(AppException):
    """权限异常（如未授权访问）"""
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_403_FORBIDDEN)

# 异常响应模型（用于FastAPI返回）
class ErrorResponse(BaseModel):
    code: int
    message: str
    detail: str = ""
    request_id: str = ""
    