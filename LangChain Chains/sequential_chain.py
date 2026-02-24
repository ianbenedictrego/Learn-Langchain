from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    repetition_penalty=1.1
)



prompt_1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template=" Generate a 5 pointer summary of the following text \n {text}",
    input_variables=['text']
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

chain = prompt_1 | model | parser | prompt_2 |model | parser

result = chain.invoke({'topic':'Cricket'})
print(result)

