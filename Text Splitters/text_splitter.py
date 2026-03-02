from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """LangChain is a framework for building applications with LLMs.
It helps with document loading, splitting, embeddings, and retrieval."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print(chunks)