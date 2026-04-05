"""使用 MultiQueryRetriever 的独立检索问答脚本。

目标：
1) 基于已有 Chroma 向量库执行检索；
2) 使用 MultiQueryRetriever 进行查询改写并合并召回；
3) 拼接上下文后调用 LLM 回答；
4) 控制台仅输出最终答案。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()
# 与现有 retriever.py 对齐的默认问题。
QUERY = "两只小鸟睡前故事说了什么"
# 与现有 retriever.py 对齐的向量库目录。
PERSIST_DIR = "./chroma_db"
# 默认召回数量。
TOP_K = 3
# OpenAI-compatible 配置（沿用仓库现有约定）。
API_KEY = os.getenv("GLM_API_KEY")
BASE_URL = os.getenv("GLM_BASE_URL")
MODEL_NAME = os.getenv("GLM_MODEL_NAME")
TEMPERATURE = 0.9
COLLECTION_NAME = "summary_index"

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手，协助用户解答问题。"),
        (
            "user",
            "根据以下提供的上下文信息，回答用户的问题：\n\n{context}\n\n用户的问题是：{input}",
        ),
    ]
)


def build_model() -> ChatOpenAI:
    """构建用于查询改写与问答生成的 LLM 客户端。"""
    return ChatOpenAI(
        api_key=SecretStr(API_KEY or ""),
        base_url=BASE_URL,
        model=MODEL_NAME or "",
        temperature=TEMPERATURE,
    )


def build_vector_retriever():
    """基于 Chroma 构建基础向量检索器（MMR）。"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    db = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K},
    )


def build_multi_query_retriever():
    """用 LLM 包装基础检索器，构建多查询检索器。"""
    model = build_model()
    retriever = build_vector_retriever()
    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=model,
    )


def answer_question(question: str) -> str:
    """执行检索与问答，并返回最终答案字符串。"""
    cleaned_question = question.strip()
    if not cleaned_question:
        return "问题不能为空。"

    model = build_model()
    multi_retriever = build_multi_query_retriever()

    try:
        docs: list[Document] = multi_retriever.invoke(cleaned_question)
    except Exception as exc:
        return f"检索失败：{exc}"

    if not docs:
        return "未检索到可用上下文，请先确认向量库中有数据。"

    context = "\n".join(doc.page_content for doc in docs)

    try:
        response = (ANSWER_PROMPT | model).invoke(
            {"context": context, "input": cleaned_question}
        )
        content = getattr(response, "content", "")
        return content if isinstance(content, str) else str(content)
    except Exception as exc:
        return f"生成答案失败：{exc}"


def main() -> None:
    """脚本入口：输出最终答案。"""
    answer = answer_question(QUERY)
    print(answer)


if __name__ == "__main__":
    main()
