# import os
# from dotenv import load_dotenv
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# import sys
# from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# load_dotenv()

# sys.stdout.reconfigure(encoding="utf-8")



# template = ChatPromptTemplate.from_messages(
#     [("system", "You are a AI that summarize the text"), 
#      ("human", "{data}" )]
# )

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
# )

# model = ChatHuggingFace(llm=llm)

###############################################################

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN",
    ""
)


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"            
)

vectorstore = Chroma(
    persist_directory="chroma-db",  
    embedding_function=embedding_model
)   

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

llm = ChatHuggingFace(llm=llm_endpoint)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """You are a helpful AI assistant that summarizes the text and provides relevant information based on the context provided. 
        Please provide concise and accurate summaries of the documents retrieved.
        Use ONLY the provided context to answer the question. If the answer is not contained within the text below, say "I could not find the answer in the document."
        """), 
        ("human", 
         """Context: {context}

         Question: {question}
         """
        )
    ]
)

print("RAG system created")

print("press 0 to exit")

while True:
    query = input("You : ")
    if query == "0":
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt = prompt.invoke({
        "context": context,
        "question": query,
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
   