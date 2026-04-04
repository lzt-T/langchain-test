"""假设性问题索引构建脚本。

流程总览：
1) 读取 PDF 文档；
2) 使用 RecursiveCharacterTextSplitter 切分 chunk；
3) 使用 LLM 为每个 chunk 生成多条“假设用户会问的问题”；
4) 将问题文本写入 Chroma 向量库；
5) 将原始 chunk 文档写入 LocalFileStore 持久化 docstore。
"""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
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
COLLECTION_NAME = "hypo_question_index"
DOCSTORE_DIR = "./docstore_hypo_chunks"
ID_KEY = "chunk_id"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
# 每个 chunk 生成的问题数量
QUESTIONS_PER_CHUNK = 4
RECREATE_COLLECTION = True


RAW_HYPO_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个检索增强系统的数据构建助手。"
            "请基于给定文本，生成用户可能会提出的问题列表。"
            "必须输出严格 JSON 数组，数组元素为字符串问题；禁止输出额外说明。",
        ),
        (
            "user",
            "请根据下面文本生成 {question_count} 条可能问题：\n\n{text}",
        ),
    ]
)


def build_question_model() -> ChatOpenAI:
    """构建生成假设问题阶段使用的 LLM 客户端。"""
    return ChatOpenAI(
        api_key=SecretStr("sk-geminixxxxx"),
        base_url="http://localhost:8000/v1",
        model="gemini-3.0-flash",
        temperature=0.2,
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """构建向量化模型，用于问题文本入库与检索。"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_pdf_docs(pdf_path: Path) -> list[Document]:
    """加载 PDF 为文档对象列表。"""
    loader = UnstructuredPDFLoader(str(pdf_path))
    return loader.load()


def build_fallback_question(text: str) -> str:
    """构建兜底问题，保证问题生成异常时索引流程不中断。"""
    trimmed = text.strip().replace("\n", " ")
    preview = trimmed[:80] if trimmed else "该内容为空"
    return f"这段内容主要讲了什么？关键信息：{preview}"


def generate_hypo_questions(model: ChatOpenAI, text: str) -> list[str]:
    """生成假设问题列表，失败时回退为单条兜底问题。"""
    trimmed = text.strip()
    if not trimmed:
        return [build_fallback_question("")]

    response = (RAW_HYPO_QUESTION_PROMPT | model).invoke(
        {"text": trimmed, "question_count": QUESTIONS_PER_CHUNK}
    )
    content = getattr(response, "content", "")
    raw_text = content if isinstance(content, str) else str(content)

    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            raise ValueError("Model output is not a JSON array.")

        questions: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    questions.append(cleaned)

        if questions:
            return questions
    except Exception:
        pass

    return [build_fallback_question(trimmed)]


def main() -> None:
    """脚本入口：构建假设性问题向量索引与 docstore。"""
    question_model = build_question_model()
    embeddings = build_embeddings()

    pdf_path = Path(PDF_PATH).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    base_docs = load_pdf_docs(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(base_docs)

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

    byte_store = LocalFileStore(str(docstore_dir))
    docstore = create_kv_docstore(byte_store)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
    )

    doc_id = pdf_path.stem
    question_docs: list[Document] = []
    question_doc_ids: list[str] = []
    generated_questions = 0

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}-chunk-{idx:04d}"
        source = str(chunk.metadata.get("source", str(pdf_path)))
        metadata = {
            "doc_id": doc_id,
            ID_KEY: chunk_id,
            "source": source,
        }

        raw_doc = Document(page_content=chunk.page_content, metadata=metadata)
        # 将 chunk 存储到 docstore 中
        retriever.docstore.mset([(chunk_id, raw_doc)])

        questions = generate_hypo_questions(question_model, chunk.page_content)
        for question_idx, question in enumerate(questions):
            question_id = f"{chunk_id}-q-{question_idx:02d}"
            question_docs.append(Document(page_content=question, metadata=metadata))
            question_doc_ids.append(question_id)
            generated_questions += 1

    #
    if question_docs:
        vectorstore.add_documents(question_docs, ids=question_doc_ids)

    stats = {
        "pdf": str(pdf_path),
        "collection": COLLECTION_NAME,
        "total_chunks": len(chunks),
        "generated_questions": generated_questions,
        "indexed_records": len(question_docs),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
