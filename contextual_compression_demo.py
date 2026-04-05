"""LangChain 上下文压缩检索 Demo（变量配置版）。

本脚本演示三种压缩器：
1) LLMChainExtractor
2) LLMChainFilter
3) EmbeddingsFilter

使用方式：
- 仅修改本文件顶部配置变量，不提供 CLI 参数。
- 运行后输出统一 JSON，便于对比不同压缩器的命中差异。
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    EmbeddingsFilter,
    LLMChainExtractor,
    LLMChainFilter,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 1) 加载环境变量
load_dotenv()

# 2) 顶部变量配置区（唯一输入接口）
MODE = "llm_filter"  # 可选：extractor | llm_filter | emb_filter
QUESTION = "故事里小兔为什么惭愧？"
TOP_K = 6
EMB_SIMILARITY_THRESHOLD = 0.76

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "summary_index"

EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
MODEL_TEMPERATURE = 0.2
SNIPPET_MAX_LEN = 200

# OpenAI-compatible 配置（沿用仓库约定）
API_KEY = os.getenv("GLM_API_KEY")
BASE_URL = os.getenv("GLM_BASE_URL")
MODEL_NAME = os.getenv("GLM_MODEL_NAME")

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的问答助手。请仅基于给定上下文回答；若上下文不足，请明确说明不确定。",
        ),
        (
            "user",
            "问题：{question}\n\n上下文：\n{context}\n\n请给出简洁、准确的答案。",
        ),
    ]
)


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量检索与 EmbeddingsFilter 使用的 embedding 模型。"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_chat_model() -> ChatOpenAI:
    """构建 LLM 压缩器使用的聊天模型。"""
    if not API_KEY or not BASE_URL or not MODEL_NAME:
        raise ValueError(
            "缺少模型环境变量，请在 .env 中配置 GLM_API_KEY / GLM_BASE_URL / GLM_MODEL_NAME。"
        )

    return ChatOpenAI(
        api_key=SecretStr(API_KEY),
        base_url=BASE_URL,
        model=MODEL_NAME,
        temperature=MODEL_TEMPERATURE,
    )


def build_compressor(mode: str, embeddings: HuggingFaceEmbeddings) -> Any:
    """根据 mode 构建对应压缩器。"""
    normalized_mode = mode.strip().lower()

    if normalized_mode == "extractor":
        llm = build_chat_model()
        return LLMChainExtractor.from_llm(llm)

    if normalized_mode == "llm_filter":
        llm = build_chat_model()
        return LLMChainFilter.from_llm(llm)

    if normalized_mode == "emb_filter":
        return EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=EMB_SIMILARITY_THRESHOLD,
        )

    raise ValueError(
        f"不支持的 MODE: {mode}，可选值为 extractor | llm_filter | emb_filter"
    )


def build_vector_retriever(embeddings: HuggingFaceEmbeddings):
    """构建基础向量检索器。"""
    persist_dir = Path(PERSIST_DIR).resolve()
    if not persist_dir.exists():
        raise FileNotFoundError(f"向量库目录不存在: {persist_dir}")

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def to_hit_payload(records: list[Document]) -> list[dict[str, Any]]:
    """将 Document 列表转换为统一输出结构。"""
    payload: list[dict[str, Any]] = []

    for record in records:
        content = record.page_content or ""
        trimmed_content = content[:SNIPPET_MAX_LEN]

        payload.append(
            {
                "metadata": record.metadata or {},
                "page_content": trimmed_content,
            }
        )

    return payload


def build_context_from_compressed_hits(records: list[Document]) -> str:
    """将压缩命中拼接为 LLM 可读上下文。"""
    parts: list[str] = []
    for idx, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        chunk_id = str(metadata.get("chunk_id", ""))
        source = str(metadata.get("source", ""))
        header = f"[片段{idx}] chunk_id={chunk_id} source={source}"
        parts.append(header)
        parts.append(record.page_content or "")
    return "\n\n".join(parts).strip()


def generate_answer(question: str, context: str) -> str:
    """基于压缩后的上下文生成最终答案。"""
    if not context.strip():
        return "压缩后未命中可用上下文，无法基于证据回答该问题。"

    model = build_chat_model()
    response = (ANSWER_PROMPT | model).invoke(
        {"question": question, "context": context}
    )
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def main() -> None:
    """脚本入口：执行基础检索 + 上下文压缩检索，并输出结果。"""
    question = QUESTION.strip()
    if not question:
        raise ValueError("QUESTION 为空，请在顶部变量区填写查询问题。")

    embeddings = build_embeddings()
    base_retriever = build_vector_retriever(embeddings)

    # 先做一次基础召回，便于给“向量库为空或无命中”提供明确提示。
    raw_hits = base_retriever.invoke(question)
    if not raw_hits:
        raise ValueError(
            "基础检索未返回结果：请检查向量库是否为空、collection_name 是否正确，或问题是否过于偏离语料。"
        )

    compressor = build_compressor(MODE, embeddings)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor,
    )

    compressed_hits = compression_retriever.invoke(question)
    compressed_context = build_context_from_compressed_hits(compressed_hits)
    answer = generate_answer(question, compressed_context)

    result = {
        "mode": MODE,
        "question": question,
        "hit_count": len(compressed_hits),
        "compressed_hits": to_hit_payload(compressed_hits),
        "answer": answer,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
