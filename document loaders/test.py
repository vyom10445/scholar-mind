from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader

loader = TextLoader("document loaders/notes.txt" , encoding="utf-8")
docs = loader.load()

print(docs[0].page_content)