import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
vector_db = None

def create_vector_db(pdf_path="knowledge.pdf"):
    global vector_db
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    print("Vector DB Created!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    if os.path.exists("faiss_index"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing Vector DB!")
    elif os.path.exists("knowledge.pdf"):
        create_vector_db("knowledge.pdf")
    yield

app = FastAPI(lifespan=lifespan)
HTML_PAGE = """<html><head><title>DocuMind - RAG Knowledge Assistant</title><style>body{font-family:sans-serif;background:#f0f2f5;display:flex;justify-content:center;align-items:center;height:100vh;margin:0}.card{background:white;width:95%;max-width:500px;height:85vh;border-radius:15px;display:flex;flex-direction:column}.header{background:#075E54;color:white;padding:15px;text-align:center;font-weight:bold}.chat{flex:1;padding:15px;overflow-y:auto;display:flex;flex-direction:column}.msg{margin:8px 0;padding:10px 15px;border-radius:15px;max-width:80%}.user{background:#DCF8C6;align-self:flex-end}.bot{background:#eee;align-self:flex-start}.input-area{display:flex;padding:10px}input{flex:1;padding:10px;border-radius:20px;border:1px solid #ccc}button{margin-left:10px;background:#25D366;color:white;border:none;padding:10px 20px;border-radius:20px}</style></head><body><div class="card"><div class="header">🧠 DocuMind - RAG Knowledge Assistant</div><div class="chat" id="chat"><div class="msg bot">Hello! I am DocuMind. Ask me anything about the PDF in English.</div></div><div class="input-area"><input id="q" placeholder="Ask in English..."><button onclick="ask()">Send</button></div></div><script>
function typeWriter(element, text, speed=20){
  let i=0; element.innerHTML="";
  function type(){ if(i<text.length){ element.innerHTML+=text.charAt(i); i++; chat.scrollTop=chat.scrollHeight; setTimeout(type, speed);} }
  type();
}
async function ask(){let q=document.getElementById('q').value;if(!q)return;let chat=document.getElementById('chat');chat.innerHTML+=`<div class="msg user">${q}</div>`;document.getElementById('q').value='';let botDiv=document.createElement('div'); botDiv.className='msg bot'; botDiv.innerHTML='Typing...'; chat.appendChild(botDiv); chat.scrollTop=chat.scrollHeight; let res=await fetch(`/ask?q=${encodeURIComponent(q)}`);let data=await res.json();typeWriter(botDiv, data.answer || data.error, 15);}</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home(): return HTML_PAGE

@app.get("/ask")
def ask_question(q: str):

    global vector_db
    if vector_db is None: return {"error": "knowledge.pdf not found"}
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2, retries=5)
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer in English from context.")
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return {"answer": chain.invoke(q)}
