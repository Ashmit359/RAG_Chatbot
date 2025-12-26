# ==============================
# Step 1: Interface with Streamlit
# ==============================
import streamlit as st

st.set_page_config(
    page_title="Softel Assist Bot",
    page_icon="🤖",
    layout="wide"
)

# ---------- HEADER ----------
st.title("Softel Assist Bot")

st.markdown(
    """
    <div style="text-align:center; font-size:16px; color:#b0b0b0;">
        Developed by <b>Ashmit Sinha</b> | Cloud • DevOps • Generative AI Engineer
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------- SIDEBAR : DEVELOPER INFO ----------
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

st.sidebar.markdown("---")

# ---------- RESET CHAT ----------
if st.sidebar.button("Reset Chat"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# ---------- SESSION STATE ----------
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

# ---------- FILE UPLOAD ----------
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

# ==============================
# Step 2: Chat Interface
# ==============================
from streamlit_chat import message

def handle_userinput(user_question):
    with st.spinner("Generating response..."):
        result = st.session_state.conversation.invoke({"question": user_question})
        response = (
            result.content
            if hasattr(result, "content")
            else "⚠️ Softel Assist Bot is temporarily unavailable."
        )

        st.session_state.chat_history.append(f"You: {user_question}")
        st.session_state.chat_history.append(f"Bot: {response}")

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg, is_user=True, key=str(i))
        else:
            message(msg, key=str(i))

# ==============================
# Step 3: Process PDF
# ==============================
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    docs = []
    for page in pages:
        splits = splitter.split_text(page.page_content)
        for text in splits:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": "Uploaded PDF"}
                )
            )
    return docs

# ==============================
# Step 4: Embeddings
# ==============================
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embed_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=st.secrets["GEMINI_API_KEY"]
)

# ==============================
# Step 5: Qdrant Vector Store
# ==============================
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

if uploaded_file:
    with open("uploaded_temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    documents = process_pdf("uploaded_temp.pdf")

    vectorstore = QdrantVectorStore.from_documents(
        documents,
        embed_model,
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
        collection_name="Softel-Assist-Bot",
        prefer_grpc=True,
        force_recreate=True
    )

    st.session_state.processComplete = True
    st.sidebar.success("✅ PDF processed successfully")

# ==============================
# Step 6: QA Chain (Gemini)
# ==============================
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from operator import itemgetter

def get_qa_chain(vectorstore, k=3):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "Information not found in the document."

        Context:
        {context}

        Question:
        {question}
        """
    )

    llm = ChatGoogleGenerativeAI(
        api_key=st.secrets["GEMINI_API_KEY"],
        model="gemini-1.5-flash",
        temperature=0
    )

    chain = (
        RunnableMap(
            {
                "question": itemgetter("question"),
                "context": itemgetter("question") | retriever,
            }
        )
        | prompt
        | llm
    )

    return chain

# ==============================
# Step 7: User Chat
# ==============================
if st.session_state.processComplete:
    st.session_state.conversation = get_qa_chain(vectorstore)

    user_question = st.chat_input("Ask a question about the uploaded PDF")
    if user_question:
        handle_userinput(user_question)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:13px; color:#888;">
        © 2025 Ashmit Sinha • Built with Streamlit, LangChain, Qdrant & Google Gemini
    </div>
    """,
    unsafe_allow_html=True
)
