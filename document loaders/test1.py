from langchain_community.document_loaders import TextLoader

data = TextLoader("resume.txt")
docs = data.load()
print(docs)