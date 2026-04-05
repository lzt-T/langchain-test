"""混合检索索引构建脚本（单 PDF + Chroma 向量索引 + BM25 语料文件）。

流程总览：
1) 读取根目录下的 1.pdf 文件；
2) 切分为 chunks；
3) 注入稳定 chunk_id；
4) 写入 Chroma 集合；
5) 导出 BM25 语料 JSON；
6) 输出构建统计信息。
"""

import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 索引脚本配置：直接修改这些常量即可。
PDF_PATH = "1.pdf"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "hybrid_index"
RECREATE_COLLECTION = True
BM25_CORPUS_PATH = "./docstore_hybrid/bm25_corpus.json"

# 切分参数（与现有脚本同量级）。
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
    """从 PDF 文件加载 Document 列表。"""
    loader = UnstructuredPDFLoader(str(pdf_path))
    return loader.load()


def chunk_docs(base_docs: list[Document]) -> list[Document]:
    """将文档切分为 chunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(base_docs)


def add_stable_chunk_ids(chunks: list[Document]) -> list[Document]:
    """为每个 chunk 注入稳定 chunk_id，并补齐元数据。"""
    counters: dict[str, int] = {}
    result: list[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        source = str(metadata.get("source", "unknown-source"))
        index = counters.get(source, 0)
        counters[source] = index + 1

        source_key = Path(source).stem or "doc"
        chunk_id = f"{source_key}-chunk-{index:04d}"
        metadata["chunk_id"] = chunk_id
        metadata["source"] = source

        result.append(Document(page_content=chunk.page_content, metadata=metadata))
    return result


def _to_json_value(value: Any) -> Any:
    """将 metadata 值规整为可 JSON 序列化对象。"""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    return str(value)


def export_bm25_corpus(chunks: list[Document], corpus_path: Path) -> None:
    """导出 BM25 语料 JSON。"""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)

    payload: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        payload.append(
            {
                "chunk_id": str(metadata.get("chunk_id", "")),
                "page_content": chunk.page_content,
                "metadata": _to_json_value(metadata),
            }
        )

    corpus_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """脚本入口：构建混合检索所需索引与语料。"""
    pdf_path = Path(PDF_PATH).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    base_docs = load_pdf_docs(pdf_path)
    if not base_docs:
        raise ValueError(f"No content loaded from PDF: {pdf_path}")

    chunks = chunk_docs(base_docs)
    enriched_chunks = add_stable_chunk_ids(chunks)

    embeddings = build_embeddings()
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

    corpus_path = Path(BM25_CORPUS_PATH).resolve()
    export_bm25_corpus(enriched_chunks, corpus_path)

    stats = {
        "pdf_path": str(pdf_path),
        "base_docs": len(base_docs),
        "chunks": len(enriched_chunks),
        "collection": COLLECTION_NAME,
        "persist_dir": str(Path(PERSIST_DIR).resolve()),
        "bm25_corpus_path": str(corpus_path),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
