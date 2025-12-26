# ==============================
# IMPORTS & CONFIG
# ==============================
import streamlit as st

st.set_page_config(
    page_title="Softel Assist Bot",
    page_icon="🤖",
    layout="wide"
)

# ==============================
# THEME TOGGLE
# ==============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

theme_css = """
<style>
body {
    background-color: %s;
    color: %s;
}
footer {visibility: hidden;}
.stApp {
    margin-bottom: 60px;
}
.sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%%;
    background: #0e1117;
    color: #aaa;
    text-align: center;
    padding: 10px;
    font-size: 13px;
    z-index: 100;
}
</style>
""" % (
    "#0e1117" if st.session_state.theme == "dark" else "#ffffff",
    "#ffffff" if st.session_state.theme == "dark" else "#000000",
)

st.markdown(theme_css, unsafe_allow_html=True)

# ==============================
# HEADER WITH LOGO
# ==============================
col1, col2 = st.columns([1, 6])

with col1:
    st.image("logo.png", width=80)

with col2:
    st.markdown(
        """
        <h1 style="margin-bottom:0;">Softel Assist Bot</h1>
        <p style="color:#b0b0b0; margin-top:0;">
        Developed by <b>Ashmit Sinha</b> | Cloud • DevOps • GenAI Engineer
        </p>
        """,
        unsafe_allow_html=True
    )

# ==============================
# TOP ACTION BUTTONS
# ==============================
colA, colB, colC = st.columns([1, 1, 6])

with colA:
    st.button("🌗 Toggle Theme", on_click=toggle_theme)

with colB:
    if st.button("ℹ️ About Project"):
        st.session_state.show_about = True

# ==============================
# ABOUT PROJECT MODAL
# ==============================
if st.session_state.get("show_about"):
    with st.modal("About Softel Assist Bot"):
        st.markdown(
            """
            ### 🤖 Softel Assist Bot

            **Softel Assist Bot** is an intelligent document assistant built using  
            **Retrieval-Augmented Generation (RAG)** architecture.

            **🔧 Tech Stack**
            - Streamlit (UI)
            - LangChain (Orchestration)
            - Google Gemini (LLM + Embeddings)
            - Qdrant (Vector Database)

            **🎯 Use Case**
            - Ask questions from uploaded PDFs
            - Context-aware answers
            - Enterprise-ready document intelligence

            **👨‍💻 Developer**
            Ashmit Sinha  
            Cloud • DevOps • Generative AI Engineer
            """
        )
        if st.button("Close"):
            st.session_state.show_about = False

st.markdown("---")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.markdown("## ⚙️ Controls")

# Reset Chat
if st.sidebar.button("Reset Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

st.sidebar.markdown("---")

# Upload PDF
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF file",
    type="pdf"
)

st.sidebar.markdown("---")

# Developer section BELOW upload
st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.markdown(
    """
    **Ashmit Sinha**  
    Cloud • DevOps • GenAI Engineer  

    🔗 **Connect with me:**  
    - [LinkedIn](https://www.linkedin.com/in/ashmit-sinha-372115b0/)  
    - [GitHub](https://github.com/Ashmit359/)  
    - [X (Twitter)](https://x.com/sinha359)  
    - [Instagram](https://www.instagram.com/ashmit.sinha359/)
    """
)

# ==============================
# SESSION STATE
# ==============================
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = False

# ==============================
# CHAT UI
# ==============================
from streamlit_chat import message

def handle_userinput(user_question):
    with st.spinner("Generating response..."):
        result = st.session_state.conversation.invoke({"question": user_question})
        response = result.content if hasattr(result, "content") else "No answer found."

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("bot", response))

    for i, (role, text) in enumerate(st.session_state.chat_history):
        message(text, is_user=(role == "user"), key=str(i))

# ==============================
# PDF PROCESSING
# ==============================
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from operator import itemgetter

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    docs = []
    for page in pages:
        for chunk in splitter.split_text(page.page_content):
            docs.append(Document(page_content=chunk))
    return docs

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = process_pdf("temp.pdf")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        collection_name="Softel-Assist-Bot",
        force_recreate=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer using only context:\n{context}\n\nQuestion:\n{question}"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0
    )

    st.session_state.conversation = (
        RunnableMap({
            "question": itemgetter("question"),
            "context": itemgetter("question") | retriever
        }) | prompt | llm
    )

    st.session_state.processComplete = True
    st.sidebar.success("✅ PDF processed successfully")

# ==============================
# USER INPUT
# ==============================
if st.session_state.processComplete:
    question = st.chat_input("Ask a question about the PDF")
    if question:
        handle_userinput(question)

# ==============================
# STICKY FOOTER
# ==============================
st.markdown(
    """
    <div class="sticky-footer">
        © 2025 Ashmit Sinha • Built with Streamlit, LangChain, Qdrant & Google Gemini
    </div>
    """,
    unsafe_allow_html=True
)
