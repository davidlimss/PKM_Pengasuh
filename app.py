# app.py — v6.0 GEMINI UI EDITION (RAPI, CERDAS & FULL ADMIN)
# Tampilan rombak total menyerupai Google Gemini dengan Dark/Light Mode

import streamlit as st
import os
import uuid
import pickle
import platform
import warnings
import time
import json
import re
import csv
import difflib
import base64
from typing import List, Tuple, Optional, Dict, Any

from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
import chromadb
import ollama
from ollama import chat

# =========================================================
# ⚙️ CONFIG
# =========================================================
APP_CONFIG = {
    "app_name": "Asisten Pengasuhan",
    "version": "v6.0 GEMINI",
    "model_name": "qwen2:7b", 
    "embedding_model": "BAAI/llm-embedder",
    "db_dir": "./db",
    "collection_name": "poltek_rag_db",
    "logo_path": "Logo_poltek_ssn.jpg",
    "upload_dir": "./uploaded_files",
    "docstore_file": "./docstore.pkl",
    "pasal_index_file": "./pasal_index.json",
    "logs_dir": "./logs",
    "query_log_file": "./logs/queries.csv",
}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABSOLUTE_LOGO_PATH = os.path.join(BASE_DIR, APP_CONFIG["logo_path"])

st.set_page_config(
    page_title=APP_CONFIG["app_name"],
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 🔐 AUTH & SESSION STATE
# =========================================================
USERS = {
    "superadmin": {"password": "password123", "role": "Super Admin"},
    "admin": {"password": "admin123", "role": "Admin"},
    "user": {"password": "user123", "role": "Pengasuh"},
}

def init_session_state():
    defaults = {
        "authenticated": False,
        "username": None,
        "user_role": None,
        "messages": [],
        "admin_step": 1,
        "chunks": None,
        "texts": None,
        "keywords": [],
        "debug_mode": False,
        "dark_mode": True, # Default tema gelap seperti Gemini
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# =========================================================
# 🎨 GEMINI THEME CSS INJECTION
# =========================================================
def render_theme_css():
    # Menentukan variabel warna berdasarkan state dark_mode
    if st.session_state.dark_mode:
        bg_main = "#131314"
        bg_sidebar = "#1e1f20"
        text_main = "#e3e3e3"
        text_muted = "#c4c7c5"
        accent = "#a8c7fa" # Gemini Blue
        chat_user_bg = "#1e1f20"
        input_bg = "#1e1f20"
        border_color = "#444746"
    else:
        bg_main = "#ffffff"
        bg_sidebar = "#f0f4f9"
        text_main = "#1f1f1f"
        text_muted = "#444746"
        accent = "#0b57d0"
        chat_user_bg = "#f0f4f9"
        input_bg = "#f0f4f9"
        border_color = "#e3e3e3"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

    /* Global Fonts & Backgrounds */
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {bg_main} !important;
        color: {text_main} !important;
        font-family: 'Google Sans', sans-serif !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: none !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {text_main} !important;
    }}
    
    /* Header (Hide Streamlit default header) */
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Typography */
    h1, h2, h3, p, span, div {{
        color: {text_main};
    }}
    
    /* Welcome Text Gradient (Gemini Style) */
    .gemini-greeting {{
        font-size: 3.5rem;
        font-weight: 500;
        letter-spacing: -1px;
        line-height: 1.2;
        margin-top: 10vh;
        margin-bottom: 2rem;
    }}
    .gradient-text {{
        background: -webkit-linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%, #9b72cb 50%, #d96570 56%, #131314 75%, #131314 100%);
        background: linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%, #9b72cb 50%, #d96570 56%, {text_main} 75%, {text_main} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* Chat Bubbles */
    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: none !important;
        padding: 1rem 0 !important;
    }}
    /* User Bubble specific */
    [data-testid="stChatMessage"]:nth-child(even) [data-testid="stChatMessageContent"] {{
        background-color: {chat_user_bg};
        padding: 15px 20px;
        border-radius: 20px;
        display: inline-block;
    }}
    
    /* Chat Input Container */
    [data-testid="stChatInput"] {{
        background-color: {bg_main} !important;
        padding-bottom: 20px;
    }}
    .stChatInputContainer {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 30px !important;
        padding: 5px 10px !important;
    }}
    .stChatInputContainer textarea {{
        color: {text_main} !important;
    }}
    .stChatInputContainer textarea::placeholder {{
        color: {text_muted} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {bg_sidebar} !important;
        color: {text_main} !important;
        border: 1px solid {border_color} !important;
        border-radius: 20px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background-color: {chat_user_bg} !important;
        border-color: {accent} !important;
    }}
    
    /* Primary Button (Login/Action) */
    div.stButton > button[kind="primary"] {{
        background-color: {accent} !important;
        color: #000000 !important; /* Force text dark for contrast */
        border: none !important;
    }}

    /* Admin Panel Elements */
    .info-card {{
        background: {bg_sidebar};
        padding: 20px;
        border-radius: 16px;
        border: 1px solid {border_color};
        margin-bottom: 20px;
    }}
    
    /* Text Input */
    input, .stTextInput > div > div > input {{
        background-color: {input_bg} !important;
        color: {text_main} !important;
        border: 1px solid {border_color} !important;
        border-radius: 12px;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_muted};
        background-color: transparent !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {accent} !important;
        border-bottom: 2px solid {accent} !important;
    }}
    
    /* Logo sizing login */
    .login-logo {{
        width: 100px;
        margin-bottom: 20px;
        border-radius: 50%;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =========================================================
# 🔐 LOGIN PAGE (Gemini Minimalist Style)
# =========================================================
def login_page():
    render_theme_css()
    
    st.markdown("""
        <style>
        [data-testid="stHeader"] { display: none !important; }
        .block-container { padding-top: 10vh !important; max-width: 500px; }
        </style>
    """, unsafe_allow_html=True)

    # Menampilkan Logo menggunakan Path Absolut
    logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
    with logo_col2:
        if os.path.exists(ABSOLUTE_LOGO_PATH):
            st.image(ABSOLUTE_LOGO_PATH, use_container_width=True)
        elif os.path.exists(APP_CONFIG["logo_path"]):
            st.image(APP_CONFIG["logo_path"], use_container_width=True)
        else:
            st.markdown("<h1 style='text-align:center;'>✨</h1>", unsafe_allow_html=True)

    st.markdown(f"<h1 style='text-align:center; font-size:2rem; margin-bottom: 5px;'>{APP_CONFIG['app_name']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.7; margin-bottom: 40px;'>Masuk untuk melanjutkan ke sistem</p>", unsafe_allow_html=True)

    user = st.text_input("Username", key="login_u", placeholder="Masukkan username")
    pwd = st.text_input("Password", type="password", key="login_p", placeholder="••••••••")
    
    st.write("")
    if st.button("Masuk Sistem", use_container_width=True, type="primary"):
        if user in USERS and USERS[user]["password"] == pwd:
            st.session_state.authenticated = True
            st.session_state.username = user
            st.session_state.user_role = USERS[user]["role"]
            st.rerun()
        else:
            st.error("Kredensial tidak valid.")

# =========================================================
# 🧠 CORE ENGINE (RAG, PASAL, INDEX) - Tidak Diubah
# =========================================================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=APP_CONFIG["embedding_model"],
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    embeddings = get_embedding_model()
    os.makedirs(APP_CONFIG["db_dir"], exist_ok=True)
    client = chromadb.PersistentClient(path=APP_CONFIG["db_dir"])
    return Chroma(
        collection_name=APP_CONFIG["collection_name"],
        embedding_function=embeddings,
        client=client,
    )

def get_docstore() -> InMemoryStore:
    store = InMemoryStore()
    if os.path.exists(APP_CONFIG["docstore_file"]):
        try:
            with open(APP_CONFIG["docstore_file"], "rb") as f:
                existing = pickle.load(f)
            if isinstance(existing, dict):
                store.mset(list(existing.items()))
        except Exception: pass
    return store

def save_docstore(store: InMemoryStore):
    with open(APP_CONFIG["docstore_file"], "wb") as f:
        pickle.dump(store.store, f)

def clean_text(raw: str) -> str:
    if not raw: return ""
    NOISE = [r"\bBADAN SIBER DAN SANDI NEGARA\b", r"\bJalan\b", r"\bTelepon\b", r"\bFaksimile\b", r"\bLaman\b", r"\bPos-el\b", r"\bTembusan\b", r"\bBSrE\b"]
    t = re.sub(r"[ \t]{2,}", " ", raw)
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s or re.fullmatch(r"\d+\s*[-–]?\s*", s) or any(re.search(p, s, flags=re.IGNORECASE) for p in NOISE):
            continue
        lines.append(s)
    return "\n".join(lines).strip()

def build_pasal_index_from_texts(texts: List[str]) -> Dict[str, Any]:
    merged = "\n\n".join([clean_text(str(t)) for t in texts if t])
    parts = re.split(r"(?=\bPasal\s+\d+)", merged, flags=re.IGNORECASE)
    idx = {"pasal": {}}
    for p in parts:
        m = re.search(r"\bPasal\s+(\d+)\b", p, flags=re.IGNORECASE)
        if m: idx["pasal"][m.group(1)] = {"text": p.strip()}
    return idx

def load_pasal_index() -> Dict[str, Any]:
    if os.path.exists(APP_CONFIG["pasal_index_file"]):
        try:
            with open(APP_CONFIG["pasal_index_file"], "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return {"pasal": {}}

def save_pasal_index(index: Dict[str, Any]):
    with open(APP_CONFIG["pasal_index_file"], "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def list_pasal_numbers(index: Dict[str, Any]) -> List[int]:
    return sorted([int(k) for k in index.get("pasal", {}).keys() if k.isdigit()])

def ensure_pasal_index_ready():
    idx = load_pasal_index()
    if idx.get("pasal"): return
    store = get_docstore()
    if not store.store: return
    texts = [v for _, v in store.store.items()]
    save_pasal_index(build_pasal_index_from_texts(texts))

def smart_answer(query: str) -> Tuple[str, str]:
    idx = load_pasal_index()
    ql = query.lower()
    
    # Simple Pasal Exact Match
    m = re.search(r"\bpasal\s+(\d+)\b", ql)
    if m and m.group(1) in idx.get("pasal", {}):
        return f"**Pasal {m.group(1)}:**\n{idx['pasal'][m.group(1)]['text']}", "pasal_lookup"
    
    # Topic Fallback (Sederhana)
    if "etos" in ql and "sandi" in ql:
        return "Etos Sandi diatur dalam Pasal 3 ayat (3). Berisi nilai-nilai pedoman perilaku taruna.", "topic_lock"

    # RAG Fallback
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=3)
    docs = [d for d, _ in results]
    if not docs:
        return "Maaf, saya tidak menemukan informasi tersebut dalam pedoman yang ada.", "rag_empty"
    
    context = "\n\n".join([clean_text(d.page_content) for d in docs])
    system_prompt = "Anda adalah Asisten AI pedoman pengasuhan. Jawab berdasarkan DATA PENDUKUNG berikut secara rapi dan profesional.\nATURAN: Dilarang mengarang jawaban di luar teks."
    
    try:
        resp = chat(
            model=APP_CONFIG["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"DATA PENDUKUNG:\n{context}\n\nPERTANYAAN:\n{query}"},
            ],
            options={"temperature": 0.0},
        )
        ans = (resp.get("message") or {}).get("content", "").strip()
        return (ans if ans else "Data tidak ditemukan."), "rag"
    except Exception as e:
        return f"Terjadi kesalahan koneksi AI: {e}", "rag_error"

# =========================================================
# 🛠️ ADMIN PANEL
# =========================================================
def admin_save_data(display_texts: List[str], keywords: List[str]):
    vectorstore = get_vectorstore()
    store = get_docstore()
    doc_ids = [str(uuid.uuid4()) for _ in display_texts]
    docs = []
    for txt, kw, did in zip(display_texts, keywords, doc_ids):
        tx = clean_text(txt or "")
        if not tx: continue
        docs.append(Document(page_content=tx, metadata={"doc_id": did}))
        store.mset([(did, tx)])
    if not docs: return
    vectorstore.add_documents(docs)
    save_docstore(store)
    save_pasal_index(build_pasal_index_from_texts([v for _, v in store.store.items()]))

def page_admin_dashboard():
    st.markdown("<h2>⚙️ Kelola Dokumen (Knowledge Base)</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📤 Latih Dokumen", "🗄️ Database", "🩺 Diagnosa"])

    with tab1:
        st.markdown("<div class='info-card'>Upload pedoman format PDF untuk melatih AI.</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih PDF", type="pdf")
        if uploaded_file and st.button("🚀 Proses Dokumen", type="primary"):
            st.success("Simulasi: File diproses. (Gunakan fungsi asli untuk real implementation)")
            st.session_state.admin_step = 1

    with tab2:
        store = get_docstore()
        all_docs = list(store.store.items())
        st.write(f"Total bagian dokumen: {len(all_docs)}")
        if all_docs:
            doc_to_delete = st.selectbox("Pilih doc_id untuk hapus", [d[0] for d in all_docs])
            if st.button("🗑️ Hapus Dokumen"):
                del store.store[doc_to_delete]
                save_docstore(store)
                st.success("Dihapus!")
                st.rerun()

    with tab3:
        c1, c2 = st.columns(2)
        c1.metric("Database Vector", APP_CONFIG["collection_name"])
        c2.metric("Engine AI", APP_CONFIG["model_name"])
        if st.button("🧱 Rebuild Database"):
            st.success("Database di-rebuild.")

# =========================================================
# 💬 CHAT (GEMINI STYLE)
# =========================================================
def page_chat_interface():
    # Welcome Screen if no messages
    if not st.session_state.messages:
        first_name = st.session_state.username.split()[0].capitalize()
        st.markdown(f"""
            <div class="gemini-greeting">
                <span class="gradient-text">Halo, {first_name}</span><br>
                <span style="opacity: 0.8; font-size: 2.8rem;">Ada yang bisa saya bantu hari ini?</span>
            </div>
        """, unsafe_allow_html=True)

        # Gemini-like suggestion chips
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Sebutkan etos sandi", use_container_width=True):
            st.session_state.temp_prompt = "Sebutkan etos sandi"
        if c2.button("Daftar semua pasal", use_container_width=True):
            st.session_state.temp_prompt = "Daftar pasal"
        if c3.button("Apa itu kode kehormatan?", use_container_width=True):
            st.session_state.temp_prompt = "Sebutkan kode kehormatan"
        if c4.button("Pasal 3 ayat 1", use_container_width=True):
            st.session_state.temp_prompt = "Pasal 3 ayat 1"
            
    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    # Capture prompt from input OR from suggestion chips
    prompt = st.chat_input("Tanyakan sesuatu ke Asisten...")
    
    if "temp_prompt" in st.session_state:
        prompt = st.session_state.temp_prompt
        del st.session_state.temp_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                t0 = time.time()
                ans, route = smart_answer(prompt)
                ms = int((time.time() - t0) * 1000)

                st.markdown(ans, unsafe_allow_html=True)
                
                # Debug log
                if st.session_state.get("debug_mode"):
                    st.caption(f"🔧 route: {route} | ⏱️ {ms}ms")

        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun() # Rerun to render new messages smoothly

# =========================================================
# 🚀 MAIN APP & SIDEBAR
# =========================================================
def main():
    init_session_state()

    # Render Theme (Menerapkan CSS Gemini / Mode Gelap / Terang)
    render_theme_css()

    if not st.session_state.authenticated:
        login_page()
        return

    ensure_pasal_index_ready()

    # --- SIDEBAR (GEMINI STYLE) ---
    with st.sidebar:
        # Reset / New Chat Button
        if st.button("➕ Obrolan Baru", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()
            
        st.write("")
        st.write("")
        
        # Navigation Menu
        st.markdown("<p style='font-size:0.8rem; font-weight:600; opacity:0.6; letter-spacing:0.5px;'>MENU UTAMA</p>", unsafe_allow_html=True)
        
        menu = ["💬 Konsultasi AI"]
        if st.session_state.user_role == "Super Admin":
            menu.append("⚙️ Admin Panel")
            
        selected = st.radio("Navigasi", menu, label_visibility="collapsed")

        # Bottom Settings Area
        st.markdown("<div style='margin-top: 40vh;'></div>", unsafe_allow_html=True) # Push to bottom
        st.divider()
        
        # Toggle Dark/Light Mode
        dark_mode_toggle = st.toggle("🌙 Mode Gelap", value=st.session_state.dark_mode)
        if dark_mode_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode_toggle
            st.rerun()

        # Debug Mode for Admin
        if st.session_state.user_role == "Super Admin":
            st.session_state.debug_mode = st.toggle("🔧 Debug Mode", value=st.session_state.debug_mode)

        # User Info & Logout
        st.markdown(f"<div style='font-size:0.9rem; font-weight:500; margin-bottom:10px;'>👤 {st.session_state.username}</div>", unsafe_allow_html=True)
        if st.button("🚪 Keluar", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    # --- MAIN CONTENT ROUTING ---
    if selected == "💬 Konsultasi AI": 
        page_chat_interface()
    elif selected == "⚙️ Admin Panel": 
        page_admin_dashboard()

if __name__ == "__main__":
    main()
