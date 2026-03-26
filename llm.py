from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="d0191e8ead8f4222a7b1ce6a3c672a08.ItyT4WFsRYWKBWBn",
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="glm-4.7-flash",
    temperature=0.9,
    max_tokens=2048,
)

system_message = SystemMessagePromptTemplate.from_template("You are a {role}.")
human_message = HumanMessagePromptTemplate.from_template("What is {topic}?")

messages = ChatPromptTemplate.from_messages(
    [
        system_message,
        human_message,
    ]
).format_messages(role="helpful assistant", topic="Python")

response = llm.invoke(messages)

print(response.content)
