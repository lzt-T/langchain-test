"""父子索引问答脚本（ParentDocumentRetriever 版）。

流程总览：
1) 从 child 向量库召回；
2) 根据父子关联回查 parent 文档；
3) 拼接 parent 上下文交给 LLM 回答；
4) 输出 answer 与召回调试信息。
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 查询脚本配置：直接修改这些常量即可。
QUERY = "两只小鸟睡前故事说了什么"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "parent_index"
DOCSTORE_DIR = "./docstore_parent"
ID_KEY = "doc_id"
TOP_K = 5

# 与索引侧保持一致，确保检索时切分逻辑与入库一致。
PARENT_CHUNK_SIZE = 1600
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80


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
    """将检索得到的父文档拼接为上下文。"""
    parts: list[str] = []
    for idx, record in enumerate(records, start=1):
        metadata = record.metadata or {}
        header = (
            f"[父片段{idx}] {ID_KEY}={metadata.get(ID_KEY, '')} "
            f"source={metadata.get('source', '')}"
        )
        parts.append(header)
        parts.append(record.page_content)
    return "\n\n".join(parts)


def build_splitters():
    """构建 parent/child 两级切分器。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )
    return parent_splitter, child_splitter


def main() -> None:
    """脚本入口：基于父子索引执行问答。"""
    question = QUERY.strip()

    embeddings = build_embeddings()
    model = build_chat_model()
    parent_splitter, child_splitter = build_splitters()

    vectorstore = Chroma(
        persist_directory=str(Path(PERSIST_DIR).resolve()),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    byte_store = LocalFileStore(str(Path(DOCSTORE_DIR).resolve()))
    docstore = create_kv_docstore(byte_store)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K},
    )

    # 执行检索
    parent_docs = retriever.invoke(question)
    context = build_context(parent_docs)

    if not context.strip():
        answer = "未检索到可用上下文，请先构建父子索引。"
    else:
        # 生成答案
        response = (ANSWER_PROMPT | model).invoke(
            {"question": question, "context": context}
        )
        content = getattr(response, "content", "")
        answer = content if isinstance(content, str) else str(content)

    result = {
        "answer": answer,
        "retrieved_count": len(parent_docs),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
