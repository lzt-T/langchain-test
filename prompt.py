from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI


system_message = SystemMessagePromptTemplate.from_template("You are a {role}.")
human_message = HumanMessagePromptTemplate.from_template("What is {topic}?")

messages = ChatPromptTemplate.from_messages(
    [
        system_message,
        human_message,
    ]
).format_messages(role="helpful assistant", topic="Python")


print(messages)
