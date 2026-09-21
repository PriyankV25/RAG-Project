import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import sys
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")
data = PyPDFLoader("document loaders/llmresearch.pdf")
docs = data.load()
template = ChatPromptTemplate.from_messages(
    [("system", "You are a AI that summarize the text"), 
     ("human", "{data}" )]
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

prompt = template.format_messages(data=docs[0].page_content)

response = model.invoke(prompt)

print(response.content)