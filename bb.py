from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-geminixxxxx",
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
    max_tokens=2048,
)


response = llm.invoke([{"role": "user", "content": "你可以写一个python的加法代码吗？"}])
print(response.content)
