from dotenv import load_dotenv
import app.config.env as env
from pathlib import Path
import os
class Config:
    def __init__(self):
        # 1. 确认配置文件路径
        env_path = Path("app/API.env")
        print(f"📂 配置文件路径：{env_path.absolute()}")
        # 2. 检查文件是否存在
        if not env_path.exists():
            raise FileNotFoundError(f"文件不存在：{env_path.absolute()}")

        load_success = load_dotenv(dotenv_path=env_path, override=True)  # override=True：覆盖系统环境变量

        if not load_success:
            raise ValueError(f"配置文件解析失败！请检查 API.env 格式（如编码、键值对）")

    def get_api_key(self,key_type):
        if key_type=="openai":
            return env.OPENAI_API_KEY
        if key_type=="siliconflow":
            print(env.SILICONFLOW_API_KEY)
            return env.SILICONFLOW_API_KEY
        if key_type=="saliyun":
            return env.ALIYUN_DASHSCOPE_API_KEY
        else : raise NotImplementedError