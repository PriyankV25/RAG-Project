import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Use your existing Hugging Face token
os.environ["HF_TOKEN"] = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN",
    ""
)

docs = [
    Document(
        page_content="This is a sample document.",
        metadata={"source": "sample_book"}
    ),
    Document(
        page_content="This is another sample document.",
        metadata={"source": "sample2_book"}
    ), 
    Document(
        page_content="This is a third sample document.",
        metadata={"source": "sample3_book"}
    )
]

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("what is third sample document", k=2)

for r in result:
    print(r)