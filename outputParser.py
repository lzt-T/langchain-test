from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# 1. 定义期望的输出结构
class CalculationResult(BaseModel):
    answer: str = Field(description="计算过程和结果")
    status: str = Field(description="执行状态，例如 'success' 或 'error'")


# 2. 初始化模型
model = ChatOpenAI(
    api_key="sk-geminixxxxx",
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.1,  # 降低温度以获得更稳定的 JSON 输出
)

# 3. 初始化解析器并获取格式化指令
parser = JsonOutputParser(pydantic_object=CalculationResult)

# 4. 构建包含格式化指令的提示模板
# 注意：我们在模板中添加了 {format_instructions} 变量
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个 {role}。请严格按照 JSON 格式输出结果。\n{format_instructions}",
        ),
        ("human", "请计算 {topic}?"),
    ]
)

# 5. 构建链
chain = prompt | model | parser

# 6. 执行调用
# 注入 role, topic 以及自动生成的 format_instructions
result = chain.invoke(
    {
        "role": "计算机",
        "topic": "1+1",
        "format_instructions": parser.get_format_instructions(),
    }
)

print(result)
