import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN",
    ""
)


def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


def get_vectorstore():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )


def get_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


def get_llm():

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1",
        huggingfacehub_api_token=os.getenv(
            "HUGGINGFACEHUB_ACCESS_TOKEN"
        )
    )

    return ChatHuggingFace(
        llm=llm_endpoint
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that answers questions
based ONLY on the provided document context.

Rules:

1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. Give concise and accurate answers.
4. If the answer is not present in the context,
   say exactly:

"I could not find the answer in the document."

Context:

{context}
"""
        ),
        (
            "human",
            """
Question:

{question}
"""
        )
    ]
)


def ask_question(query):

    retriever = get_retriever()

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    llm = get_llm()

    response = llm.invoke(final_prompt)

    return response.content, docs