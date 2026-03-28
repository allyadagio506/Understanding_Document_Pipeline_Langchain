#Length Based Text Splitter

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text = TextLoader("text.txt",encoding="utf-8")
docs = text.load()

splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)

result = splitter.split_documents(docs)

print(result[0].page_content)

#Text-Structure Based Text Splitter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text = TextLoader("text.txt",encoding="utf-8")
docs = text.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0,separators=["\n\n","\n",".","!","?"," ", ""])
result = splitter.split_documents(docs)
print(result[0].page_content)