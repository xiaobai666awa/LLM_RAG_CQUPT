import app.core.model as md


def base_chat_test(sender):
    mdt=md.get_chat_model(type="siliconflow",name="Qwen/Qwen3-8B")
    print(mdt.invoke(sender))
    return mdt.invoke(sender)

print(base_chat_test("你好"))



