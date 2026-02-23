from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

def build_prompt(text):
    word_count = len(text.split())

    if word_count < 150:
        return f"""
You are a professional research assistant.
Provide a concise summary in 5-6 sentences.
Do NOT include reasoning steps.

Text:
{text}
"""

    elif word_count < 800:
        return f"""
You are a research analyst.
Provide a structured summary with:

1. Problem
2. Proposed Solution
3. Key Contributions
4. Impact

Keep it clear and complete.
Do NOT show thinking process.

Text:
{text}
"""

    else:
        return f"""
You are a senior research reviewer.
Provide a detailed structured summary with:

- Background
- Methodology
- Core Innovations
- Experimental Results
- Impact and Future Work

Ensure full sentences and a proper conclusion.
Do NOT include internal reasoning.

Text:
{text}
"""

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=1.3,
    # max_new_tokens= 700,
    repetition_penalty=1.1
)

model =  ChatHuggingFace(llm = llm)

st.header("Research Tool")
user_input = st.text_input("Enter your Prompt")


if st.button("Summarize"):
    if not user_input.strip():
        st.warning("Please enter text.")
    else:
        prompt = build_prompt(user_input)

        with st.spinner("Generating summary..."):
            result = model.invoke(prompt)
        st.write(result.content)


