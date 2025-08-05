import json
import app.test.base_chat_with_model as tset_md

class Conversation:
    def __init__(self, model,sender,receiver):
        self.model = model
        self.sender = sender
        self.receiver = receiver
    def to_dict(self):
        return {"model":self.model,"sender":self.sender,"receiver":self.receiver}
    def to_json(self):
        return json.dumps(self.to_dict())
    def conversation(self):
