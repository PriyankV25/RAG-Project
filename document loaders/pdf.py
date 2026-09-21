from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter


data = PyPDFLoader("PriyankResumeDevOps.pdf")
docs = data.load()

splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)

chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks)


