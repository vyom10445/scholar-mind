from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI , OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma


embedding = OpenAIEmbeddings()

#retreive data from chromadb
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

#create retrieval
retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwards = {
        "k" : 4 ,
        "fetch_k" : 10,
        "lambda_mult" : 0.5
    }
)


llm = ChatOpenAI(model="gpt-5.4-nano-2026-03-17")


#prompt template 

prompt = ChatPromptTemplate.from_messages([
    (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
])

#just for terminal purpose (would remove)

print("RAG System created")

print("press 0 to exit")

while True:
    query = input("You: ")
    if query == "0":
        break
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )


    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")