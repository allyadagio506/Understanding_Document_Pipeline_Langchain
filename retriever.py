#Retriever Wikipedia

from langchain_community.retrievers import WikipediaRetriever
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

api = os.getenv("OPENAI_API_KEY")

retriever = WikipediaRetriever(top_k_results=3,lang="en")
query = "history of saladin"

docs = retriever.invoke(query)

for i,docs in enumerate(docs):
    print(f"\n------Result{i+1}------")
    print(f"Content--\n{docs.page_content}--")

#Vector Store Retriever

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="langchain_docs"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

query = "What is Chroma used for?"
results = retriever.invoke(query)

for i,res in enumerate(results):
    print(f"\n------Result{i+1}------")
    print(f"Content--\n{res.page_content}--")




