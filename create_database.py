import os
from dotenv import load_dotenv
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
data = PyPDFLoader("document loaders/deep-learning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)

os.environ["HF_TOKEN"] = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN",
    ""
)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"            
)

vectorstore = Chroma.from_documents(
    documents=chunks,   
    embedding=embedding_model,
    persist_directory="chroma-db"
)