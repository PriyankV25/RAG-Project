from langchain_community.document_loaders import WebBaseLoader
import sys
sys.stdout.reconfigure(encoding="utf-8")

url = "https://www.apple.com/in/macbook-pro/"

data = WebBaseLoader(url)
docs = data.load()
print(len(docs))
print(docs[0].page_content)