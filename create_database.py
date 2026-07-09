#load pdf 
#split into chunks 
#create embeddings
#store into vectorstoredb

from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI


loader = PyPDFLoader("document loaders/deeplearning.pdf")
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)

chunks = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

