import os
from langchain.document_loaders import WebBaseLoader

# loader = WebBaseLoader('https://n.news.naver.com/mnews/article/092/0002307222?sid=105')

# data = loader.load()
# print(data[0].page_content)


from langchain.document_loaders import UnstructuredURLLoader

urls = [
    'https://n.news.naver.com/mnews/article/092/0002307222?sid=105',
    # 'https://n.news.naver.com/mnews/article/052/0001944792?sid=105',
]

loader = UnstructuredURLLoader(urls=urls)

data = loader.load()
print(data)