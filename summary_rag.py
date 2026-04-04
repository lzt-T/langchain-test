"""摘要检索 RAG 查询脚本（MultiVectorRetriever 版）。

流程总览：
1) 在摘要向量库召回摘要；
2) MultiVectorRetriever 按 chunk_id 回查 LocalFileStore 中的原文 chunk；
3) 拼接原文上下文并交给 LLM 作答；
4) 输出 answer/retrieved_chunks/retrieval_debug。
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import (
    MultiVectorRetriever,
)
from langchain_classic.storage import (
    LocalFileStore,
    create_kv_docstore,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 查询脚本配置：直接修改这些变量即可。
QUERY = "两只小鸟睡前故事说了什么"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "summary_index"
DOCSTORE_DIR = "./docstore_raw_chunks"
TOP_K = 5
ID_KEY = "chunk_id"


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的RAG问答助手。请优先依据提供的原文上下文回答，不确定时明确说明。",
        ),
        (
            "user",
            "问题：{question}\n\n原文上下文：\n{context}\n\n请给出简洁、准确的回答。",
        ),
    ]
)


def build_answer_model() -> ChatOpenAI:
    """构建问答阶段使用的 LLM 客户端。"""
    return ChatOpenAI(
        api_key=SecretStr("sk-geminixxxxx"),
        base_url="http://localhost:8000/v1",
        model="gemini-3.0-flash",
        temperature=0.6,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量检索使用的 embedding 模型。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_context(records: list[Document]) -> str:
    """将多个原文记录拼成可读上下文，保留片段头信息便于追溯。"""
    parts: list[str] = []
    for idx, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        header = (
            f"[片段{idx}] chunk_id={metadata.get('chunk_id', '')} "
            f"source={metadata.get('source', '')}"
        )
        parts.append(header)
        parts.append(record.page_content)
    return "\n\n".join(parts)


def main() -> None:
    """脚本入口：执行摘要检索并基于原文上下文生成答案。"""
    question = QUERY.strip()

    model = build_answer_model()
    embeddings = build_embeddings()
    # 向量数据库
    vectorstore = Chroma(
        persist_directory=str(Path(PERSIST_DIR).resolve()),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    byte_store = LocalFileStore(str(Path(DOCSTORE_DIR).resolve()))
    docstore = create_kv_docstore(byte_store)
    # 检索器
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K},
    )

    raw_docs = retriever.invoke(question)

    context = build_context(raw_docs)
    # 基于上下文生成答案
    if not context.strip():
        answer = "未检索到可用上下文，请先构建索引或检查 docstore 数据。"
    else:
        response = (ANSWER_PROMPT | model).invoke(
            {"question": question, "context": context}
        )
        answer = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    result = {
        "answer": answer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
