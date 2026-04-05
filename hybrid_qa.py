"""混合检索问答脚本（BM25 + 向量 + EnsembleRetriever）。

流程总览：
1) 加载 .env 中模型配置；
2) 连接 Chroma 向量检索器；
3) 从 BM25 语料 JSON 还原 BM25 检索器；
4) 用 EnsembleRetriever 融合两路召回；
5) 组装上下文并调用 LLM 回答；
6) 输出 question/answer/hits 调试信息。
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 1) 加载 .env 文件
load_dotenv()

# 查询脚本配置：直接修改这些常量即可。
QUESTION = "故事"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "hybrid_index"
BM25_CORPUS_PATH = "./docstore_hybrid/bm25_corpus.json"

TOP_K_BM25 = 5
TOP_K_DENSE = 5
ENSEMBLE_WEIGHTS = [0.4, 0.6]
RRF_C = 60
HIT_SNIPPET_MAX_LEN = 120

# OpenAI-compatible 配置（沿用仓库现有约定）。
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
            "问题：{question}\n\n原文上下文：\n{context}\n\n请给出简洁、准确的回答。",
        ),
    ]
)


def build_chat_model() -> ChatOpenAI:
    """构建问答模型客户端。"""
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


def load_bm25_docs(corpus_path: Path) -> list[Document]:
    """从 JSON 文件加载 BM25 文档。"""
    if not corpus_path.exists():
        raise FileNotFoundError(f"BM25 corpus not found: {corpus_path}")

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    docs: list[Document] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        page_content = str(item.get("page_content", "")).strip()
        if not page_content:
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_id = str(item.get("chunk_id") or metadata.get("chunk_id") or "")
        if chunk_id:
            metadata["chunk_id"] = chunk_id
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def build_hybrid_retriever(bm25_docs: list[Document]) -> EnsembleRetriever:
    """构建 BM25 + 向量混合检索器。"""
    bm25 = BM25Retriever.from_documents(bm25_docs)
    bm25.k = TOP_K_BM25

    embeddings = build_embeddings()
    vectorstore = Chroma(
        persist_directory=str(Path(PERSIST_DIR).resolve()),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    dense = vectorstore.as_retriever(search_kwargs={"k": TOP_K_DENSE})

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=ENSEMBLE_WEIGHTS,
        c=RRF_C,
        id_key="chunk_id",
    )


def build_context(records: list[Document]) -> str:
    """将检索结果拼接为问答上下文。"""
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
    """脚本入口：执行混合检索并回答问题。"""
    question = QUESTION.strip()
    if not question:
        raise ValueError("QUESTION is empty")

    bm25_docs = load_bm25_docs(Path(BM25_CORPUS_PATH).resolve())
    if not bm25_docs:
        raise ValueError(f"BM25 corpus is empty: {Path(BM25_CORPUS_PATH).resolve()}")

    hybrid_retriever = build_hybrid_retriever(bm25_docs)
    records = hybrid_retriever.invoke(question)

    context = build_context(records)
    if not context.strip():
        answer = "未检索到可用上下文，请先运行 hybrid_index.py 构建索引。"
    else:
        model = build_chat_model()
        response = (ANSWER_PROMPT | model).invoke(
            {"question": question, "context": context}
        )
        content = getattr(response, "content", "")
        answer = content if isinstance(content, str) else str(content)

    result = {
        "question": question,
        "answer": answer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
