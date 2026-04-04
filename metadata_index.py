"""元数据索引构建脚本（单 PDF + Chroma）。

流程总览：
1) 加载单个 PDF；
2) 切分为 chunks；
3) 为每个 chunk 注入统一 metadata；
4) 写入 Chroma；
5) 输出构建统计信息。
"""

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 索引脚本配置：直接修改这些常量即可。
PDF_PATH = "1.pdf"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "metadata_index"
RECREATE_COLLECTION = True

# 文档级 metadata：全部统一为字符串，便于向量库过滤。
DOC_ID = "doc-001"
DOC_TITLE = "两只小鸟睡前故事"
CATEGORY = "story"
CREATED_AT = "2026-04-04"

# 切分参数。
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量化模型，用于文档入库与检索。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf_docs(pdf_path: Path) -> list[Document]:
    """加载 PDF 为文档对象列表。"""
    loader = UnstructuredPDFLoader(str(pdf_path))
    return loader.load()


def chunk_docs(base_docs: list[Document]) -> list[Document]:
    """将文档切分为 chunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(base_docs)


def with_metadata(chunks: list[Document], source_path: Path) -> list[Document]:
    """为每个 chunk 注入统一 metadata。"""
    enriched: list[Document] = []
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata.update(
            {
                "doc_id": str(DOC_ID),
                "doc_title": str(DOC_TITLE),
                "category": str(CATEGORY),
                "source": str(source_path),
                "created_at": str(CREATED_AT),
                "chunk_id": f"{DOC_ID}-chunk-{index:04d}",
            }
        )
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched


def main() -> None:
    """脚本入口：构建带 metadata 的向量索引。"""
    pdf_path = Path(PDF_PATH).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    embeddings = build_embeddings()
    base_docs = load_pdf_docs(pdf_path)
    chunks = chunk_docs(base_docs)
    enriched_chunks = with_metadata(chunks, pdf_path)

    vectorstore = Chroma(
        persist_directory=str(Path(PERSIST_DIR).resolve()),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    if RECREATE_COLLECTION:
        try:
            vectorstore.delete_collection()
        except Exception:
            pass
        vectorstore = Chroma(
            persist_directory=str(Path(PERSIST_DIR).resolve()),
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )

    vectorstore.add_documents(enriched_chunks)

    stats = {
        "pdf": str(pdf_path),
        "collection": COLLECTION_NAME,
        "persist_dir": str(Path(PERSIST_DIR).resolve()),
        "base_docs": len(base_docs),
        "chunks": len(enriched_chunks),
        "doc_id": DOC_ID,
        "category": CATEGORY,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
