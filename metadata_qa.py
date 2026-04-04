"""元数据问答脚本（SelfQueryRetriever + Chroma）。

流程总览：
1) 连接已有 Chroma 集合；
2) 使用 SelfQueryRetriever 自动解析问题中的 metadata 条件；
3) 执行向量检索并组装上下文；
4) 基于上下文生成答案；
5) 输出 answer 与检索调试信息。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 1. 加载 .env 文件
load_dotenv()

# 查询脚本配置：直接修改这些常量即可。
QUESTION = "总结一下两只小鸟睡前故事"
TOP_K = 5
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "metadata_index"
DOCUMENT_CONTENT_DESCRIPTION = "中文文档片段，主要是故事和知识内容"

# 和现有脚本保持一致的 OpenAI-compatible 配置风格。
API_KEY = os.getenv("GLM_API_KEY")
BASE_URL = os.getenv("GLM_BASE_URL")
MODEL_NAME = os.getenv("GLM_MODEL_NAME")
TEMPERATURE = 0.2

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的RAG问答助手。请优先依据提供的原文上下文回答，不确定时明确说明。",
        ),
        (
            "user",
            "问题：{question}\\n\\n原文上下文：\\n{context}\\n\\n请给出简洁、准确的回答。",
        ),
    ]
)


def build_chat_model() -> ChatOpenAI:
    """构建问答与 SelfQuery 共用的 LLM 客户端。"""
    return ChatOpenAI(
        api_key=SecretStr(API_KEY or ""),
        base_url=BASE_URL,
        model=MODEL_NAME or "",
        temperature=TEMPERATURE,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量检索使用的 embedding 模型。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_context(records: list[Document]) -> str:
    """将检索文档拼接为上下文。"""
    parts: list[str] = []
    for idx, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        header = (
            f"[片段{idx}] doc_id={metadata.get('doc_id', '')} "
            f"category={metadata.get('category', '')} source={metadata.get('source', '')}"
        )
        parts.append(header)
        parts.append(record.page_content)
    return "\\n\\n".join(parts)


def build_metadata_fields() -> list[AttributeInfo]:
    """定义 SelfQuery 可用的 metadata 字段。"""
    return [
        AttributeInfo(
            name="doc_id",
            description="文档唯一标识，例如 doc-001。",
            type="string",
        ),
        AttributeInfo(
            name="doc_title",
            description="文档标题，例如 两只小鸟睡前故事。",
            type="string",
        ),
        AttributeInfo(
            name="category",
            description="文档类别，例如 story、medical、policy。",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="文档来源路径。",
            type="string",
        ),
        AttributeInfo(
            name="created_at",
            description="文档创建日期字符串，例如 2026-04-04。",
            type="string",
        ),
    ]


def main() -> None:
    """脚本入口：执行 SelfQuery 元数据检索并回答问题。"""
    question = QUESTION.strip()

    embeddings = build_embeddings()
    model = build_chat_model()

    vectorstore = Chroma(
        persist_directory=str(Path(PERSIST_DIR).resolve()),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    metadata_fields = build_metadata_fields()
    retriever = SelfQueryRetriever.from_llm(
        llm=model,
        vectorstore=vectorstore,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=metadata_fields,
        search_kwargs={"k": TOP_K},
        enable_limit=True,
        verbose=True,
    )

    docs = retriever.invoke(question)

    context = build_context(docs)

    if not context.strip():
        answer = "未检索到可用上下文，请先构建索引，或调整问题中的 metadata 条件。"
    else:
        response = (ANSWER_PROMPT | model).invoke(
            {"question": question, "context": context}
        )
        content = getattr(response, "content", "")
        answer = content if isinstance(content, str) else str(content)

    result = {
        "answer": answer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
