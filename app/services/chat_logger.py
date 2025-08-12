import json
import os

class ChatLogger:
    @staticmethod
    def create_conversation(cid):
        data={
            "cid": cid,
            "history": [],
            "model":{
                "type":"siliconflow",
                "name":"Qwen/Qwen3-8B"
            }
        }
        with open(f"d../ata/Chat/{cid}","w") as f:
            json.dump(data,f)

        with open(f"d../ata/Chat/conversation.json","r") as f:
            conversation = json.load(f)
        if conversation:
            conversation["conversation_list"] .append({"cid": cid,"conversation_name":f"conversation_{cid}"})
        else:
            conversation = {"conversation_list":[]}
            conversation["conversation_list"].append({"cid": cid,"conversation_name":f"conversation_{cid}"})
        with open(f"../data/Chat/conversation.json","w") as f:
            json.dump(conversation,f)


        return "success"
    @staticmethod
    def append(cid:str, ctx:dict):
        with open(f"../data/Chat/{cid}","r") as f:
            data = json.load(f)
        data["history"].append(ctx)
        with open(f"../data/Chat/{cid}","w") as f:
            json.dump(data,f)
        return "success"

    @staticmethod
    def list_conversations():
        with open(f"../data/Chat/conversation.json","r") as f:
            conversation = json.load(f)
        return conversation["conversation_list"]
    @staticmethod
    def load(cid:str):
        with open(f"../data/Chat/{cid}.json","r") as f:
            data = json.load(f)
        return data['history']
    @staticmethod
    def delete(cid:str):
        os.remove(f"../data/Chat/{cid}.json")
        with open("../data/Chat/conversation.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "conversation_list" in data and isinstance(data["conversation_list"], list):
            # 用列表推导式过滤掉指定 cid 的字典
            data["conversation_list"] = [
                conv for conv in data["conversation_list"] if conv.get("cid") != cid
            ]

        # 写回 JSON 文件
        with open("../data/Chat/conversation.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    @staticmethod
    def get_model(self,cid):
        with open(f"../data/Chat/{cid}.json","r") as f:
            data = json.load(f)
        return data["model"]
    @staticmethod
    def update_model(cid:str, ctx:dict):
        with open(f"../data/Chat/{cid}.json","w") as f:
            data = json.load(f)
        data["model"] = ctx
        with open(f"../data/Chat/{cid}.json","w") as f:
            json.dump(data,f)
            return "success"



