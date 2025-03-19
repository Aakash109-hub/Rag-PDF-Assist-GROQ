import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
if os.getenv("RENDER") is None:
    from dotenv import load_dotenv
    load_dotenv()

# API Key Verification
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ API key is missing! Set it in `.env` or Render environment variables.")
    st.info("💡 Create a `.env` file in the root directory and add: `GROQ_API_KEY=your_api_key_here`")
    st.stop()
else:
    logging.info(f"✅ API Key Loaded: {GROQ_API_KEY[:4]}********")

# Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Failed to extract text from {pdf.name}: {e}")
    return text if text else "No text extracted."

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    return text_splitter.split_text(text)

# Store text embeddings in FAISS
@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(text_chunks, embedding=embeddings)

# Create the conversational chain
@st.cache_resource
def get_conversational_chain(vector_store):
    prompt_template = """
    Answer the question based on the context below. If the context doesn't contain the answer, say 'Answer is not available in the context.'

    Context: {context}

    Question: {question}

    Answer:
    """
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever(), chain_type="stuff", chain_type_kwargs={"prompt": prompt})

# Handle user input
def user_input(user_question):
    if "vector_store" not in st.session_state or not st.session_state.vector_store:
        st.warning("⚠️ No processed PDF data available. Please upload and process PDFs first.")
        return

    chain = get_conversational_chain(st.session_state.vector_store)
    response = chain.run(user_question)
    st.subheader("🤖 AI Response")
    st.write(response)

# Main App
def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="📄")
    st.header("📄 Chat with Your PDF Using GROQ AI")

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("📥 Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("✅ PDF Processing Complete!")

    # User Input for Queries
    user_question = st.text_input("💬 Ask a Question from the PDFs:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()

