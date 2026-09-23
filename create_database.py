# import os
# from dotenv import load_dotenv
# import sys
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# load_dotenv()

# sys.stdout.reconfigure(encoding="utf-8")
# data = PyPDFLoader("document loaders/deep-learning.pdf")
# docs = data.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
# chunks = splitter.split_documents(docs)

# os.environ["HF_TOKEN"] = os.getenv(
#     "HUGGINGFACEHUB_ACCESS_TOKEN",
#     ""
# )


# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"            
# )

# vectorstore = Chroma.from_documents(
#     documents=chunks,   
#     embedding=embedding_model,
#     persist_directory="chroma-db"
# )

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_vector_database(pdf_path, persist_directory="chroma-db"):

    print(f"Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Remove old database if necessary before creating a new book database.
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print("Vector database created successfully.")

    return vectorstore