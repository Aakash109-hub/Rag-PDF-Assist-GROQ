import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

if os.getenv("RENDER") is None:
    from dotenv import load_dotenv
    load_dotenv()

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ API key is missing! Set it in `.env` or Render environment variables.")
    st.info("💡 Create a `.env` file in the root directory and add: `GROQ_API_KEY=your_api_key_here`")
    st.stop()
else:
    logging.info(f"✅ API Key Loaded: {GROQ_API_KEY[:4]}********")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, session_id):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index_{session_id}")

def get_conversational_chain():
    prompt_template = """
    Answer the question based on the context below. If the context doesn't contain the answer, say "Answer is not available in the context."

    Context: {context}

    Question: {question}

    Answer:
    """
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, session_id):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    st.warning("Loading FAISS index with deserialization enabled. Ensure the file is from a trusted source.")
    new_db = FAISS.load_local(
        f"faiss_index_{session_id}",
        embeddings,
        allow_dangerous_deserialization=True  # Enable deserialization
    )
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", page_icon="💁")
    st.header("Chat with PDF using GROQ💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question, st.session_state.session_id)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, st.session_state.session_id)
                st.success("Done")

if __name__ == "__main__":
    main()
