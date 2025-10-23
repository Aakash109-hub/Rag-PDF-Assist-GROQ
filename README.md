# 📄 RAG-Based PDF QA Chatbot

## 🚀 Overview
This project is a **Retrieval-Augmented Generation (RAG) based chatbot** that allows users to query PDFs using **natural language**. The chatbot extracts text from uploaded PDF files, processes it into vector embeddings, and provides relevant answers using an **LLM (Large Language Model) powered by Groq API**

## 🛠 Tech Stack
- **Python**
- **LangChain** (for retrieval and chaining)
- **FAISS** (vector database for similarity search)
- **Hugging Face Transformers** (sentence embeddings)
- **Streamlit** (for UI)
- **Groq LLM** (for natural language responses)
- **PyPDF2** (for extracting text from PDFs)
- **dotenv** (for environment variable management)

## ✨ Features
✅ **Upload multiple PDFs** and extract content automatically  
✅ **Text chunking and embedding** using FAISS for fast retrieval  
✅ **Query the PDF content** and receive **LLM-generated responses**  
✅ **Custom prompt template** for context-aware answers  
✅ **Interactive UI with Streamlit** for ease of use  
✅ **Secure API key management** with environment variables  

## 📌 Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Aakash109-hub/Rag-PDF-Assist-GROQ.git
cd rag-pdf-chatbot
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add:
```
GROQ_API_KEY=your_api_key_here
```

### 5️⃣ Run the Application
```bash
streamlit run app.py
```

## 🔧 Usage
1️⃣ **Upload one or more PDFs** via the sidebar.  
2️⃣ Click on **Submit & Process** to extract and store embeddings.  
3️⃣ Type your question in the **input box** and get answers from the PDFs.  

## 🖼️ Screenshots
![Screenshot 2025-03-20 125520](https://github.com/user-attachments/assets/fe7a457f-25d3-4186-b526-bcc76d54a5f7)


## 📜 File Structure
```
├── app.py  # Main Streamlit App
├── requirements.txt  # Dependencies
├── .env  # API Keys (not included in repo)
├── README.md  # Project Documentation
├── assets/  # Store images/screenshots
```

## 🔗 Links
- **Live Demo**: [https://groq-chat-ak.onrender.com]

## 📬 Contact
If you have any questions, feel free to reach out!

🔗 LinkedIn: [(https://www.linkedin.com/in/aakashgayke109/)]  
