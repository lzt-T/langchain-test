from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

model = ChatOpenAI(
    api_key=SecretStr("sk-geminixxxxx"),
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
)

# 1. 定义包含占位符的 Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个得力的助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | model

# 2. 模拟内存存储（实际应用中可对接 Redis/SQL）
store = {}


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 3. 使用 RunnableWithMessageHistory 包装,创建对话链
with_message_history = RunnableWithMessageHistory(
    chain,  # 基础对话链
    get_session_history,  # 获取/创建会话历史的函数
    input_messages_key="input",  # 输入消息的键
    history_messages_key="history",  # 历史消息的键，与propmt中的一致
)

# 4. 调用（通过 config 指定 session_id）
config: RunnableConfig = {"configurable": {"session_id": "user_001"}}
response = with_message_history.invoke({"input": "你好，我是 lzt"}, config=config)
print(response.content)
response = with_message_history.invoke({"input": "我叫什么"}, config=config)
print(response.content)
