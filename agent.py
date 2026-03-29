from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    api_key="sk-geminixxxxx",
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
)


agent = create_agent(
    model=model,
    tools=[],  # 没有工具就传空列表
    system_prompt="你是一个助手。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "1*2等于几"}]})

print(result["messages"][-1].content)
