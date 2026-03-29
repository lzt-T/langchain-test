from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

model = ChatOpenAI(
    api_key=SecretStr("sk-geminixxxxx"),
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
)


# 创建工具
@tool
def multiply(a: int, b: int) -> int:
    """将两个整数相乘。"""  # 这个是必须的，工具的描述
    return a * b


agent = create_agent(
    model=model,
    tools=[multiply],
    system_prompt="你是一个助手，涉及计算时优先调用工具。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "1*2等于几"}]})

print(result["messages"][-1].content)
