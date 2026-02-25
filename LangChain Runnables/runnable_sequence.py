from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Fetch API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

prompt_1  = PromptTemplate(
    template="Write a joke about  {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']

)

model = ChatGoogleGenerativeAI(
    model = "gemini-3-flash-preview",
)

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

result = chain.invoke({'topic':'AI'})

print(result)