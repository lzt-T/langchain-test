import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import transformers
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    api_key="sk-geminixxxxx",
    base_url="http://localhost:8000/v1",
    model="gemini-3.0-flash",
    temperature=0.9,
    max_tokens=2048,
)

transformers.utils.logging.set_verbosity_error()
# 获取当前文件的绝对路径
root_dir = os.path.dirname(os.path.abspath(__file__))
# 文本数据文件所在目录
document_dir = os.path.join(root_dir, "documents", "test.txt")

# 创建文本加载器
loader = TextLoader(document_dir, encoding="utf-8")
# 加载文档
documents = loader.load()

# 分块大小
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 将文档分块
chunks = text_splitter.split_documents(documents)

# 加载Chroma数据库,保存到本地磁盘中，方便后续使用
# save_db = Chroma.from_documents(
#     chunks,
#     HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5"),
#     persist_directory="./chroma_db",
# )

# 数据库客户端，建立链接
client = chromadb.Client()

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 从磁盘加载数据库
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

input = "苹果是什么"

# 先将 db 转化为 retriever，并指定搜索类型为 mmr
retriever = db.as_retriever(
    # 这里指定搜索类型为 mmr，mmr 是一种基于相关性和多样性的搜索算法，可以返回更相关且多样化的结果
    search_type="mmr",
    #  mmr 搜索时需要指定 k 的值，表示返回多少条结果
    search_kwargs={"k": 3},
)

#  调用 retriever 获取查询结果
results = retriever.invoke(input)

# 将查询结果拼接成字符串
context = "\n".join([result.page_content for result in results])

# 构建提示词
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个有帮助的助手，协助用户解答问题。"),
        (
            "user",
            "根据以下提供的上下文信息，回答用户的问题：\n\n{context}\n\n用户的问题是：{input}",
        ),
    ]
)

# 调用语言模型生成答案
response = llm.invoke(prompt.format(context=context, input=input))
print(response.content)
