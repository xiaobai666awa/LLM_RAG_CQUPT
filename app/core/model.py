from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.confg import Config

def get_chat_model(type,name):
    try:

        if type =="openai":
                return ChatOpenAI(model=name,api_key=Config.get_api_key("openai"))
        elif type=="siliconflow":
            return ChatOpenAI(base_url="https://api.siliconflow.cn/v1",api_key=Config.get_api_key("siliconflow"),model=name)
        else:
            return "Unknown model type"
    except:
        return "have no this model"
def get_embedding_model():
    return OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-8B",api_key=Config.get_api_key("siliconflow"),base_url="https://api.siliconflow.cn/v1")
