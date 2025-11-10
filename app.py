import streamlit as st
import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ API key is missing! Set it in `.env` or environment variables.")
    st.info("💡 Example: `GROQ_API_KEY=your_api_key_here`")
    st.stop()

# --------------------------
# Persistent Storage Path
# --------------------------
# Use /data on Render (Persistent Disk), fallback to local when running locally
PERSIST_DIR = "/data/indexes" if os.path.exists("/data") else "indexes"
os.makedirs(PERSIST_DIR, exist_ok=True)

# --------------------------
# Helper Functions
# --------------------------
def load_and_split(file_path):
    """Load PDF and split into text chunks."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
    return text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(texts, embeddings, index_path):
    """Build and save FAISS vector store."""
    embedding_size = len(embeddings.embed_query("hello"))
    index = faiss.IndexFlatL2(embedding_size)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(texts)
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)
    return vector_store

def load_vector_store(index_path, embeddings):
    """Load FAISS index safely."""
    try:
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

def rag_answer(vector_store, query):
    """Generate answer using retrieved context and Groq model."""
    results = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""
You are a helpful assistant. Use the context below to answer clearly.

Context:
{context}

Question:
{query}

Answer:
"""
    model = ChatGroq(
        model="compound-beta",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    response = model.invoke(prompt)
    return response.content

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="📘 PDF RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# --------------------------
# Custom CSS
# --------------------------
st.markdown("""
    <style>
        body { background-color: #0e1117; color: #f0f2f6; }
        .main-title {
            text-align: center; font-size: 40px; font-weight: bold;
            background: linear-gradient(90deg, #00B4DB, #0083B0);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subheader {
            text-align: center; font-size: 18px; color: #9ba3b4;
            margin-bottom: 20px;
        }
        .chat-bubble-user {
            background-color: #0078D4; color: white; padding: 10px 15px;
            border-radius: 15px; margin: 5px 0; max-width: 75%;
            align-self: flex-end;
        }
        .chat-bubble-bot {
            background-color: #2e2e2e; color: #e6e6e6; padding: 10px 15px;
            border-radius: 15px; margin: 5px 0; max-width: 75%;
            align-self: flex-start;
        }
        .stSpinner > div > div {
            border-top-color: #00B4DB !important;
        }
        .css-1y4p8pa, .stMarkdown { line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Page Header
# --------------------------
st.markdown("<div class='main-title'>🤖 Chat with your PDF</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Powered by LangChain + Groq + HuggingFace Embeddings</div>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📘 About the App")
st.sidebar.info("""
This is a **PDF RAG Chatbot** built using:
- 🧠 LangChain  
- ⚡ Groq API  
- 🔍 FAISS + HuggingFace Embeddings  
- 🎨 Streamlit  

Upload your PDF and ask questions directly from its content!
""")

# --------------------------
# Main Layout
# --------------------------
col1, col2 = st.columns([1.2, 2.8])

with col1:
    st.markdown("### 📂 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        base_name = os.path.splitext(uploaded_file.name)[0]
        index_path = os.path.join(PERSIST_DIR, f"{base_name}_index")

        with st.spinner("🔍 Loading or creating FAISS index..."):
            vector_store = load_vector_store(index_path, embeddings)
            if vector_store:
                st.success(f"✅ Loaded existing index for **{base_name}**")
            else:
                st.info(f"⚙️ Creating index for **{base_name}**...")
                texts = load_and_split(temp_path)
                vector_store = build_vector_store(texts, embeddings, index_path)
                st.success("🎉 Index created successfully!")

        st.session_state.vector_store = vector_store

with col2:
    st.markdown("### 💬 Chat Interface")
    if "vector_store" not in st.session_state:
        st.info("👈 Please upload a PDF first.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.chat_input("Ask something about your PDF...")

        if query:
            with st.spinner("🤔 Thinking..."):
                answer = rag_answer(st.session_state.vector_store, query)

            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("bot", answer))

        for role, text in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"<div class='chat-bubble-user'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-bot'>{text}</div>", unsafe_allow_html=True)


