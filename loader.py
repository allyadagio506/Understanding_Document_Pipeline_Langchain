#Text loader

from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gpt-4o-mini"
)

loader = TextLoader("text.txt",encoding="utf-8")

quiz = PromptTemplate(
    template="write the summmary for the following {poem}",
    input_variables=["poem"]
)



parser = StrOutputParser(

)

docs = loader.load()

print(docs[0])

chain = quiz|model|parser

result = chain.invoke({"poem":docs[0].page_content})
print(result)