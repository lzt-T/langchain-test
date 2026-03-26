from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 1. 定义示例集
examples = [
    {"input": "破釜沉舟", "output": "不给自己留退路，像极了还没写完代码就敢点上线按钮的我。"},
    {"input": "守株待兔", "output": "拒绝主动加班，坐在工位上等产品经理自己取消需求。"}
]

# 2. 配置示例的格式化方式
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="词语: {input}\n释义: {output}"
)

# 3. 创建 FewShotPromptTemplate
dynamic_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请仿照以下风格，为给出的词语提供幽默的解释：", # 提示词前缀
    suffix="词语: {user_input}\n释义:",             # 提示词后缀
    input_variables=["user_input"]
)

# 生成最终的 Prompt
print(dynamic_prompt.format(user_input="卧薪尝胆"))