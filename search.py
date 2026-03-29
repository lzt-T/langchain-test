import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

model = ChatOpenAI(
    api_key="sk-geminixxxxx",
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
)

# 初始化搜索封装器
search_wrapper = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_KEY"))

# 创建搜索工具 必须显式传入 api_wrapper
search_tool = TavilySearchResults(api_wrapper=search_wrapper, k=3)

tools = [search_tool]

chain = model.bind_tools(tools)

result = chain.invoke("今天多少度")
print(result)
