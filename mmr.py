import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

api = os.getenv("OPENAI_API_KEY")


docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="mmr_docs"
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3, "lambda_mult": 0.4}
)

query = "What is langchain?"
results = retriever.invoke(query)

for i,docs in enumerate(results):
    print(f"\n------Result{i+1}------")
    print(f"Content--\n{docs.page_content}--")