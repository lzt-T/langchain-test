"""摘要索引构建脚本。

流程总览：
1) 读取 PDF 文档；
2) 使用 RecursiveCharacterTextSplitter 切分 chunk；
3) 使用 LLM 为每个 chunk 生成摘要；
4) 将摘要写入 Chroma 向量库；
5) 将原始 chunk 文档写入 LocalFileStore 持久化 docstore，供 MultiVectorRetriever 回查。
"""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import (
    MultiVectorRetriever,
)
from langchain_classic.storage import (
    LocalFileStore,
    create_kv_docstore,
)
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

# 构建脚本配置：直接修改这些变量即可。
PDF_PATH = "1.pdf"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "summary_index"
# 原文 chunk 持久化存储目录
DOCSTORE_DIR = "./docstore_raw_chunks"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RECREATE_COLLECTION = True
ID_KEY = "chunk_id"


# 摘要提示词：要求输出紧凑、便于检索的短摘要。
RAW_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个文档摘要助手。请将输入内容压缩为便于向量检索的摘要，保留关键名词、数字、结论和上下文限定。",
        ),
        (
            "user",
            "请用不超过120字总结下面文本，输出纯文本：\n\n{text}",
        ),
    ]
)


def build_summary_model() -> ChatOpenAI:
    """构建摘要阶段使用的 LLM 客户端。"""
    return ChatOpenAI(
        api_key=SecretStr("sk-geminixxxxx"),
        base_url="http://localhost:8000/v1",
        model="gemini-3.0-flash",
        temperature=0.2,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量化模型，用于摘要文本入库与检索。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def summarize_chunk(model: ChatOpenAI, text: str) -> str:
    """对单个 chunk 做摘要，返回纯文本摘要。"""
    trimmed = text.strip()
    if not trimmed:
        return "空内容"

    response = (RAW_SUMMARY_PROMPT | model).invoke({"text": trimmed})
    content = getattr(response, "content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()

    # 当模型返回异常结构时，退化为截断原文，保证索引流程不断。
    return trimmed[:120]


def load_pdf_docs(pdf_path: Path) -> list[Document]:
    """加载 PDF 为文档对象列表。"""
    loader = UnstructuredPDFLoader(str(pdf_path))
    return loader.load()


def main() -> None:
    """脚本入口：构建摘要向量索引与 docstore。"""
    model = build_summary_model()
    embeddings = build_embeddings()

    pdf_path = Path(PDF_PATH).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    base_docs = load_pdf_docs(pdf_path)
    # 分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(base_docs)

    persist_dir = Path(PERSIST_DIR).resolve()
    # 原文 chunk 持久化存储目录
    docstore_dir = Path(DOCSTORE_DIR).resolve()
    # 向量数据库本体
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    if RECREATE_COLLECTION:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        if docstore_dir.exists():
            shutil.rmtree(docstore_dir)

    byte_store = LocalFileStore(str(docstore_dir))
    docstore = create_kv_docstore(byte_store)
    # 检索器封装层
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
    )

    doc_id = pdf_path.stem
    summary_docs: list[Document] = []
    summary_ids: list[str] = []
    docstore_written_count = 0

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}-chunk-{idx:04d}"
        source = str(chunk.metadata.get("source", str(pdf_path)))

        summary_text = summarize_chunk(model, chunk.page_content)

        metadata = {
            "doc_id": doc_id,
            ID_KEY: chunk_id,
            "source": source,
        }

        summary_docs.append(Document(page_content=summary_text, metadata=metadata))
        summary_ids.append(chunk_id)

        raw_doc = Document(page_content=chunk.page_content, metadata=metadata)
        # 即时写入 docstore，避免 raw_doc_pairs 常驻内存。
        retriever.docstore.mset([(chunk_id, raw_doc)])
        docstore_written_count += 1

    if summary_docs:
        vectorstore.add_documents(summary_docs, ids=summary_ids)

    stats = {
        "pdf": str(pdf_path),
        "collection": COLLECTION_NAME,
        "docstore_dir": str(docstore_dir),
        "total_chunks": len(chunks),
        "indexed_records": len(summary_docs),
        "docstore_records": docstore_written_count,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
