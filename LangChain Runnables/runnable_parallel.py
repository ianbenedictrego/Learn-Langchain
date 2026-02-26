from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Fetch API key
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(
    model = "gemini-3-flash-preview"
)

parser =  StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})


result = parallel_chain.invoke({'topic':'AI'})

print(result)
