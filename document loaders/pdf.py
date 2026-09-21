from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("PriyankResumeDevOps.pdf")
docs = data.load()
print(docs)
