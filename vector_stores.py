from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

api = os.getenv("OPENAI_API_KEY")

doc1 = Document(
    page_content="Babar Azam is one of the most consistent batsmen in modern cricket. Known for his elegant stroke play and calm leadership, he has captained Pakistan in all formats.",
    metadata={"team": "Pakistan"}
)

doc2 = Document(
    page_content="Shaheen Afridi is a world-class fast bowler known for his deadly swing and ability to take early wickets. He has been a key player for Pakistan in all formats.",
    metadata={"team": "Pakistan"}
)

doc3 = Document(
    page_content="Mohammad Rizwan is a reliable wicketkeeper-batsman known for his consistency and hardworking attitude. He plays crucial innings for Pakistan in pressure situations.",
    metadata={"team": "Pakistan"}
)

doc4 = Document(
    page_content="Shadab Khan is a talented all-rounder who contributes with both bat and ball. His leg-spin bowling and aggressive batting make him a valuable asset.",
    metadata={"team": "Pakistan"}
)

doc5 = Document(
    page_content="Wasim Akram is one of the greatest fast bowlers in cricket history. Known as the Sultan of Swing, he played a major role in Pakistan's success, including the 1992 World Cup.",
    metadata={"team": "Pakistan"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma_db",
    collection_name="cricket_players"
)

vector_store.add_documents(docs)

v = vector_store.get(include=["embeddings"])
print(v)

similarity_results = vector_store.similarity_search(
    query="Who is the best fast bowler in Pakistan?",
    k=1,
    
)
print(similarity_results)