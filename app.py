import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Fortinet GPT Assistant", layout="centered")
st.title("🔐 Fortinet GPT Assistant")
st.write("Ask me anything about Fortinet configurations, best practices, or solutions.")

# PDF loading function
@st.cache_resource
def load_vectorstore():
    data_dir = "data/pdfs"
    db_dir = "data/db"

    # Load all PDFs in data/pdfs/
    all_docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(data_dir, filename))
            all_docs.extend(loader.load())

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Embed and store
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_dir)
    vectordb.persist()
    return vectordb

# Load or build the vector database
try:
    vectordb = Chroma(persist_directory="data/db", embedding_function=OpenAIEmbeddings())
except:
    vectordb = load_vectorstore()

# Build RAG QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=vectordb.as_retriever()
)

# Chat interface
query = st.text_input("💬 Enter your question:")
if query:
    with st.spinner("Thinking..."):
        try:
            answer = qa_chain.run(query)
            st.markdown(f"**Answer:**\n{answer}")
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
