from app.services.chat_logger import ChatLogger


class ConversationManager(ChatLogger):
    def get_context(self,cid,history_limit):
        data=ChatLogger().load(cid)
        if len(data)<history_limit:
            return data
        else:
            return data[-history_limit:]