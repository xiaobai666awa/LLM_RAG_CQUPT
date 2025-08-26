from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import Config

def get_chat_model(type,name):
        conf=Config()

        api_key = conf.get_api_key(key_type=type)
        print(f"📌 当前模型类型：{type}")
        print(f"📌 读取到的密钥前缀：{api_key[:5]}..." if api_key else "📌 未读取到密钥")

        if type =="openai":
                return ChatOpenAI(model=name,api_key=conf.get_api_key(key_type="openai"))
        elif type=="siliconflow":
            return ChatOpenAI(base_url="https://api.siliconflow.cn/v1",api_key=conf.get_api_key(key_type="siliconflow"),model=name)

        else:
            return "Unknown model type"
def get_embedding_model():
    conf=Config()
    print(conf.get_api_key(key_type="siliconflow"))
    return OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-8B",api_key=conf.get_api_key(key_type="siliconflow"),base_url="https://api.siliconflow.cn/v1")
