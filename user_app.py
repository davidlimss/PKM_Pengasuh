import streamlit as st
import os
import warnings
import platform
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import chat

# --- CONFIGURATION ---
OLLAMA_MODEL_NAME = "qwen2:1.5b"
DB_DIR = "./db"
COLLECTION_NAME = "multi_modal_rag"
LOGO_PATH = "logo_ssn.png"

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Asisten Digital Pengasuhan",
    page_icon="🤖",
    layout="wide",
)

# --- CUSTOM CSS FOR BLUE & WHITE THEME ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f8faff;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #002d72;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Header styling */
    .main-header {
        color: #002d72;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        border-bottom: 2px solid #002d72;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* Input styling */
    .stChatInputContainer {
        border-top: 1px solid #e0e6ed;
        background-color: white;
    }
    
    /* Status indicator */
    .status-box {
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #0056b3;
        background-color: #e7f1ff;
        color: #002d72;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- INITIALIZATION ---
@st.cache_resource
def load_rag_components():
    # Setup PATH for Windows binaries if they exist locally
    if platform.system() == "Windows":
        poppler_path = os.path.join(os.getcwd(), "poppler-lib", "poppler-24.08.0", "Library", "bin")
        tesseract_path = r"C:\Program Files\Tesseract-OCR"
        if os.path.exists(poppler_path) or os.path.exists(tesseract_path):
            os.environ["PATH"] += os.pathsep + poppler_path + os.pathsep + tesseract_path

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
    
    # Load vectorstore
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    return vectorstore

def ask_knowledge_base(vectorstore, question: str) -> str:
    # 1. Similarity Search
    docs = vectorstore.similarity_search(question, k=6)
    if not docs:
        return "Mohon maaf, saya tidak menemukan informasi yang relevan di dalam dokumen pengasuhan."

    # 2. Context preparation
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # 3. System Prompt
    messages = [
        {
            "role": "system",
            "content": (
                "Anda adalah asisten AI khusus untuk Sistem Pengenalan Pengasuhan di institusi ini.\n"
                "Tugas Anda: Menjawab pertanyaan berdasarkan CONTEXT yang diberikan.\n"
                "Aturan ketat:\n"
                "1. Jawab HANYA berdasarkan isi CONTEXT.\n"
                "2. Jika jawaban tidak ada di CONTEXT, katakan: 'Mohon maaf, informasi tersebut tidak ditemukan dalam pedoman pengasuhan yang ada.'\n"
                "3. Gunakan bahasa Indonesia yang formal, sopan, dan profesional.\n"
                "4. Sebutkan pasal, ayat, atau poin aturan jika tersedia di dalam teks."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nPERTANYAAN:\n{question}",
        },
    ]

    # 4. Ollama Call
    try:
        result = chat(
            model=OLLAMA_MODEL_NAME,
            messages=messages,
            options={"temperature": 0.1},
        )
        return result["message"]["content"]
    except Exception as e:
        return f"Terjadi kesalahan saat menghubungi otak AI: {str(e)}"

# --- UI LAYOUT ---
def main():
    # Sidebar
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=150)
        st.markdown("### Asisten Digital Pengasuhan")
        st.markdown("---")
        st.markdown("""
        **Tentang Sistem:**
        Aplikasi ini dirancang untuk memudahkan akses informasi mengenai pedoman, aturan, dan prosedur pengasuhan berdasarkan dokumen resmi.
        
        **Cara Penggunaan:**
        1. Ketik pertanyaan Anda di kolom chat.
        2. AI akan mencari informasi di dokumen resmi.
        3. Jawaban akan tampil secara otomatis.
        """)
        st.info("Status: Database Ready ✅")

    # Main Area
    st.markdown('<h1 class="main-header">Politeknik Siber dan Sandi Negara</h1>', unsafe_allow_html=True)
    st.subheader("💡 Asisten Digital Pengasuhan")

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Tanyakan sesuatu tentang pengasuhan..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🔍 Sedang mencari informasi...")
            
            # Initialization check
            vectorstore = load_rag_components()
            
            response = ask_knowledge_base(vectorstore, prompt)
            message_placeholder.markdown(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
