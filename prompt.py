from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

system_message = SystemMessagePromptTemplate.from_template("You are a {role}.")
human_message = HumanMessagePromptTemplate.from_template("What is {topic}?")

prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        human_message,
    ]
)

formatted_prompt = prompt.format_messages(role="helpful assistant", topic="Python")


print(formatted_prompt)
