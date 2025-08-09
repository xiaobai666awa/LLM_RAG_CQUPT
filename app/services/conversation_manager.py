from app.services.chat_logger import ChatLogger


class ConversationManager(ChatLogger):
    def get_context(self,cid,history_limit):
        data=super(ChatLogger().load(cid))
        return data[history_limit:]