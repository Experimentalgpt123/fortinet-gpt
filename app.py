import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Set page title
st.set_page_config(page_title="Fortinet GPT", layout="centered")
st.title("🔐 Fortinet GPT Assistant")
st.write("Ask me anything about Fortinet configurations, best practices, or solutions.")

# Load your existing vector DB
embedding = OpenAIEmbeddings()
db = Chroma(persist_directory="./db", embedding_function=embedding)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=db.as_retriever()
)

# Chat UI
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.markdown(f"**Answer:**\n{answer}")
