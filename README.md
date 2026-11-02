# 🧠 DocuMind - RAG Knowledge Assistant

A professional RAG (Retrieval-Augmented Generation) based AI assistant that answers questions from your own PDF documents using FastAPI, LangChain, FAISS & Google Gemini.

### 🚀 Live Demo
> Note: This project is hosted on Render's free tier, so the server sleeps after inactivity. First request may take 30-40 seconds to wake up.
👉 [Live Link]

### ✨ Features
-  Answers ONLY from your PDF (No Hallucination)
-  ChatGPT-like Word-by-Word Typing Effect
-  FastAPI Backend + FAISS Vector Database
-  Google Gemini 1.5 Flash LLM
-  Clean & Responsive UI

### 🛠️ Tech Stack
- **Backend:** FastAPI, Python
- **RAG Framework:** LangChain
- **Vector DB:** FAISS
- **Embeddings & LLM:** Google Generative AI (Gemini)
- **Frontend:** HTML, CSS, JavaScript

### ⚙️ How to Run Locally
1. Clone the repo:
   git clone https://github.com/usmanAISystem/DocuMind-RAG-Assistant.git

2. Install requirements:
   `pip install -r requirements.txt`

3. Create `.env` file and add your key:
   `GOOGLE_API_KEY=your_api_key_here`

4. Run the server:
   `uvicorn main:app --reload`

### 👨‍💻 Author
**Muhammad Usman** 
