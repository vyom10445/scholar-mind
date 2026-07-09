from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

model = ChatMistralAI(model = "mistral-small-2506")

loader = PyPDFLoader("document loaders/deeplearning.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap=200)



prompt_temp = ChatPromptTemplate([
    ("system","you are an AI that summarizes text"),
    ("human","{data}")
])

prompt = prompt_temp.invoke({"data": docs[0].page_content})


result = model.invoke(prompt)

print(result.content)