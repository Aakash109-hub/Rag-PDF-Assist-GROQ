import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# Load environment variables
if os.getenv("RENDER") is None:
    from dotenv import load_dotenv
    load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("⚠️ Groq API key is missing! Set it in a `.env` file.")
    st.stop()

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""  # Avoid NoneType errors
            text += page_text
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Store text embeddings in FAISS
def get_vector_store(text_chunks):
    return FAISS.from_texts(text_chunks, embedding=embeddings)

# Create the conversational chain
def get_conversational_chain(vector_store):
    prompt_template = """
    Answer the question using the provided context. If the answer isn't available, say 
    "The answer is not available in the context." Do NOT attempt to answer incorrectly.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever(), chain_type="stuff", chain_type_kwargs={"prompt": prompt})

# Handle user queries (without storing chat history)
def user_input(user_question, chain):
    response = chain.run(user_question)
    return response

# Main App
def main():
    st.set_page_config(page_title="SmartPDF AI")
    st.header("📄 SmartPDF AI: Your Personal Document Assistant")

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        
        if st.button("📥 Process Documents"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("✅ PDF Processing Complete!")

    # User input field
    user_question = st.text_input("💬 Ask a Question from the PDFs:")

    if user_question and st.session_state.vector_store:
        chain = get_conversational_chain(st.session_state.vector_store)
        response = user_input(user_question, chain)

        # Display response without storing chat history
        st.subheader("🤖 AI Response")
        st.write(response)

if __name__ == "__main__":
    main()
