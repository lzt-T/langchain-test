import os

# 从 langchain_community 导入 PDF 加载器
from langchain_community.document_loaders import UnstructuredPDFLoader

# 获取当前工作目录
root = os.getcwd()

# 拼接 PDF 文件的完整路径（假设文件名为 1.pdf）
document_path = os.path.join(root, "1.pdf")

# 初始化 PyPDFLoader 实例
loader = UnstructuredPDFLoader(document_path)

# 加载 PDF 内容，将其拆分为页面列表
# 每个页面对象包含 page_content（文本）和 metadata（页码等元数据）
pages = loader.load()

# 打印第一页的内容
if len(pages) > 0:
    print(pages[0].page_content)
else:
    print("未能加载到页面内容，请检查 PDF 文件是否为空或路径是否正确。")
