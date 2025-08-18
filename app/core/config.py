import json
from typing import Optional

import app.config.env as env

class Config:
    def get_api_key(self,key_type):
        if key_type=="openai":
            return env.OPENAI_API_KEY
        if key_type=="siliconflow":
            print(env.SILICONFLOW_API_KEY)
            return env.SILICONFLOW_API_KEY
        if key_type=="saliyun":
            return env.ALIYUN_DASHSCOPE_API_KEY
        else : raise NotImplementedError