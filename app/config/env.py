import os
from dotenv import load_dotenv
from pathlib import Path

# 加载.env文件（开发环境）
env_path = Path(__file__).parent.parent / "API.env"
if env_path.exists():
    load_dotenv(env_path)


def get_env(key: str, default: str = None, required: bool = False) -> str:
    """
    获取环境变量
    :param key: 环境变量名
    :param default: 默认值（可选）
    :param required: 是否为必填项，若为True且无值则抛出异常
    :return: 环境变量值
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"环境变量 {key} 未设置，请在.env文件中配置")
    return value


# 阿里云大模型API密钥（必填）
ALIYUN_DASHSCOPE_API_KEY = get_env("ALIYUN_DASHSCOPE_API_KEY", required=True)

# OpenAI API密钥（可选，若使用GPT模型则需配置）
OPENAI_API_KEY = get_env("OPENAI_API_KEY")

# Redis连接配置（用于对话记忆）
REDIS_HOST = get_env("REDIS_HOST", "localhost")
REDIS_PORT = get_env("REDIS_PORT", "6379")
REDIS_PASSWORD = get_env("REDIS_PASSWORD", "")

# 日志文件路径
LOG_PATH = get_env("LOG_PATH", "logs/app.log")
    