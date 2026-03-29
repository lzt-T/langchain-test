from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key="sk-geminixxxxx",
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.1,  # 降低温度以获得更稳定的 JSON 输出
)


# 自定义chain
@chain
def my_custom_step(input_text: str):
    # 你的自定义逻辑，比如在 Windows 路径下查找文件
    return f"处理后的结果: {input_text}"


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

chain = prompt | model | my_custom_step

result = chain.invoke({"input": "你是谁"})
print(result)
