from langchain_huggingface import HuggingFaceEmbeddings

# 1. 初始化模型
# model_name 可以是本地路径，也可以是 Hugging Face 上的 ID
model_name = "BAAI/bge-large-zh-v1.5"
model_kwargs = {"device": "cpu"}  # 如果你有英伟达显卡，可以改为 'cuda'
encode_kwargs = {"normalize_embeddings": True}  # 归一化，使得向量余弦相似度计算更准确

embeddings = HuggingFaceEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

texts = ["你好，世界！", "今天天气不错。"]

query = "今天天气怎么样？"

# 2. 生成文本的向量表示
vectors = embeddings.embed_documents(texts)

# 3. 生成查询的向量表示
query_vector = embeddings.embed_query(query)


print(vectors)

print(query_vector)
