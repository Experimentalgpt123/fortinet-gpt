import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

# Load your OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

# Load PDF(s)
loader = PyMuPDFLoader("Fortinet_Admin_Guide.pdf")  # Replace with your filename
docs = loader.load()

# Chunk the content
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store in DB
embedding = OpenAIEmbeddings()
db = Chroma.from_documents(chunks, embedding, persist_directory="./db")
db.persist()

# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=db.as_retriever()
)

# Ask questions
while True:
    query = input("Ask me something about Fortinet: ")
    if query.lower() in ["exit", "quit"]:
        break
    answer = qa.run(query)
    print(f"\nAnswer: {answer}")
