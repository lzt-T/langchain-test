import os

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

model = ChatOpenAI(
    api_key=SecretStr("sk-geminixxxxx"),
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
)


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 从磁盘加载数据库
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

input = "苹果是什么"

# 先将 db 转化为 retriever，并指定搜索类型为 mmr
retriever = db.as_retriever(
    # 这里指定搜索类型为 mmr，mmr 是一种基于相关性和多样性的搜索算法，可以返回更相关且多样化的结果
    search_type="mmr",
    #  mmr 搜索时需要指定 k 的值，表示返回多少条结果
    search_kwargs={"k": 3},
)

#  调用 retriever 获取查询结果
results = retriever.invoke(input)

# 将查询结果拼接成字符串
context = "\n".join([result.page_content for result in results])
# 构建提示词
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手，协助用户解答问题。"),
        (
            "user",
            "根据以下提供的上下文信息，回答用户的问题：\n\n{context}\n\n用户的问题是：{input}",
        ),
    ]
)

chain = prompt | model

# 调用语言模型生成答案
response = chain.invoke({"context": context, "input": input})

print(response.content)
