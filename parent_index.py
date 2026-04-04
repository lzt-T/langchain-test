"""父子索引构建脚本（ParentDocumentRetriever 版）。

流程总览：
1) 加载单个 PDF；
2) 使用 parent_splitter / child_splitter 进行父子分层切分；
3) 将 child 文档写入 Chroma；
4) 将 parent 文档写入 LocalFileStore；
5) 输出索引统计信息。
"""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 索引脚本配置：直接修改这些常量即可。
PDF_PATH = "1.pdf"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "parent_index"
DOCSTORE_DIR = "./docstore_parent"
ID_KEY = "doc_id"
RECREATE_COLLECTION = True

# 父子切分参数：父块大、子块小，兼顾召回精度与上下文完整性。
PARENT_CHUNK_SIZE = 1600
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 80


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量化模型，用于 child 文档写入与检索。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf_docs(pdf_path: Path) -> list[Document]:
    """加载 PDF 为文档对象列表。"""
    loader = UnstructuredPDFLoader(str(pdf_path))
    return loader.load()


def build_splitters() -> tuple[
    RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitter
]:
    """构建 parent/child 两级切分器。"""
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
    """脚本入口：构建父子索引。"""
    pdf_path = Path(PDF_PATH).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    embeddings = build_embeddings()
    parent_splitter, child_splitter = build_splitters()
    base_docs = load_pdf_docs(pdf_path)

    persist_dir = Path(PERSIST_DIR).resolve()
    docstore_dir = Path(DOCSTORE_DIR).resolve()

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

    # 文档持久化
    byte_store = LocalFileStore(str(docstore_dir))
    docstore = create_kv_docstore(byte_store)

    # 构建 ParentDocumentRetriever 检索器
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key=ID_KEY,
    )
    retriever.add_documents(base_docs)

    stats = {
        "pdf": str(pdf_path),
        "collection": COLLECTION_NAME,
        "persist_dir": str(persist_dir),
        "docstore_dir": str(docstore_dir),
        "base_docs": len(base_docs),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
