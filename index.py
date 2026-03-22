import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_BASE_URL"),
)

system_prompt = "我是智能机器人LZT，我可以帮助你解决一些问题。"


def main( question: str ):
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.6,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    st.title("智能机器人LZT")
    st.write("我可以帮助你解决一些问题。")
    st.write("请输入你的问题：")
    question = st.text_input("问题：")
    if st.button("提交"):
        response = main(question)
        st.write(response)
    # main()
