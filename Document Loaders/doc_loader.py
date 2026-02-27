from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

prompt = PromptTemplate(
    template="Generate a summary of the document on {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(
    model = "gemini-3-flash-preview"
)

parser = StrOutputParser()

loader = TextLoader('E:/Langchain/Document Loaders/cars.txt', encoding= 'utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'topic':docs[0].page_content})

print(result)


