# import os
# # os.environ["OPENAI_API_KEY"]="sk-dxpsdazxgbpfenbnncezdtnyappzaczpxkjjdmuuhnnvsxhp"
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter
# from llama_index.core import SimpleDirectoryReader
# raw_documents = TextLoader("./data/测试.txt").load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# documents = text_splitter.split_documents(raw_documents)
# texts = text_splitter.split_documents(documents)
# embed_model = OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-8B",base_url="https://api.siliconflow.cn/v1")
# vectorstore = FAISS.from_documents(texts, embed_model)
# retriever = vectorstore.as_retriever()
# docs = retriever.invoke("测试")
# print(docs)
#  # filepath: example.py
