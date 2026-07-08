from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(model = "mistral-small-2506")

loader = TextLoader("document loaders/notes.txt" , encoding="utf-8")
docs = loader.load()

prompt_temp = ChatPromptTemplate([
    ("system","you are an AI that summarizes text"),
    ("human","{data}")
])

prompt = prompt_temp.invoke({"data": docs[0].page_content})


result = model.invoke(prompt)

print(result.content)