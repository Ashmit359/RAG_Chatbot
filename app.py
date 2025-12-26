import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import tempfile
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Softel AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ------------------ THEME SWITCH ------------------
theme = st.toggle("🌙 Dark Mode")

if theme:
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# ------------------ HEADER ------------------
col1, col2 = st.columns([1, 8])

with col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=80)

with col2:
    st.markdown("""
    ## 🤖 Softel AI Chatbot  
    **AI-powered PDF Question Answering System**
    """)

# ------------------ ABOUT MODAL ------------------
with st.expander("ℹ️ About Project"):
    st.markdown("""
    **Softel AI Chatbot** is a Retrieval-Augmented Generation (RAG) system that:
    - Reads PDF documents
    - Converts text into embeddings
    - Stores them in Qdrant
    - Answers questions using Google Gemini

    **Tech Stack**
    - Streamlit
    - LangChain
    - Qdrant
    - HuggingFace Embeddings
    - Google Gemini
    """)

# ------------------ DEVELOPER SECTION ------------------
st.markdown("""
### 👨‍💻 Developer  
**Ashmit Sinha**  
Cloud • DevOps • GenAI Engineer  

🔗 **Connect with me:**  
- [LinkedIn](https://www.linkedin.com/in/ashmit-sinha-372115b0/)  
- [GitHub](https://github.com/Ashmit359/)  
- [X (Twitter)](https://x.com/sinha359)  
- [Instagram](https://www.instagram.com/ashmit.sinha359/)
""")

st.divider()

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "📄 Upload a PDF file",
    type=["pdf"],
    help="Limit 200MB per file • PDF only"
)

# ------------------ EMBEDDINGS (FREE) ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ QDRANT CLIENT ------------------
qdrant_client = QdrantClient(
    path="./qdrant_data"
)

# ------------------ PROCESS PDF ------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        client=qdrant_client,
        collection_name="pdf_rag"
    )

    st.success("✅ PDF processed successfully")

    # ------------------ LLM (GEMINI CHAT ONLY) ------------------
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    # ------------------ CHAT ------------------
    query = st.text_input("💬 Ask a question from the PDF")

    if query:
        with st.spinner("Thinking..."):
            response = qa.run(query)
            st.markdown("### 🤖 Answer")
            st.write(response)

# ------------------ STICKY FOOTER ------------------
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
© 2025 Ashmit Sinha • Built with Streamlit, LangChain, Qdrant & Google Gemini
</div>
""", unsafe_allow_html=True)
