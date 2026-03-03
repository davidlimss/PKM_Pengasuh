# app.py — v5.1 PERFECT BALANCE (RAPI, CERDAS & FULL ADMIN)
# Ini adalah versi final hasil kombinasi fitur super rapi (Code 2) dengan layout admin yang lengkap (Code 1).
# 
# Peningkatan yang didapat:
# 1. Tampilan Chat sangat rapi: Ada Rujukan / Pasal / Ayat / Huruf secara spesifik.
# 2. Fitur Admin utuh: Bisa membedah file PDF secara utuh, menyimpan dan Rebuild Database secara full!
# 3. Model qwen2:1.5b dipaksa patuh sama dokumen. 

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
    "app_name": "Sistem Cerdas Pengasuhan",
    "version": "v5.1 PERFECT",
    "model_name": "qwen2:1.5b", 
    "embedding_model": "BAAI/llm-embedder",
    "db_dir": "./db",
    "collection_name": "poltek_rag_db",
    "logo_path": "logo_ssn.png",
    "upload_dir": "./uploaded_files",
    "docstore_file": "./docstore.pkl",
    "pasal_index_file": "./pasal_index.json",
    "logs_dir": "./logs",
    "query_log_file": "./logs/queries.csv",
}

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# =========================================================
# 🎨 UI SETUP & CSS
# =========================================================
st.set_page_config(
    page_title=APP_CONFIG["app_name"],
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
:root{
  --primary:#0B1E3D; --primary2:#162B54; --accent:#00B4D8;
  --bg:#F4F6F9; --white:#fff; --muted:#64748b;
}
.stApp{ background:var(--bg); font-family:'Inter',sans-serif; }
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,var(--primary) 0%,var(--primary2) 100%);
  border-right:1px solid rgba(255,255,255,.06);
}
[data-testid="stSidebar"] *{ color:#fff !important; }
.main-header{
  font-size:2.1rem; font-weight:800; color:var(--primary);
  border-bottom:4px solid var(--accent); padding-bottom:10px;
}
.info-card{
  background:var(--white); padding:22px; border-radius:16px;
  border:1px solid #eef2f6; box-shadow:0 4px 14px rgba(0,0,0,.03);
}
.user-badge{
  background:rgba(255,255,255,.10); padding:14px; border-radius:12px;
  border:1px solid rgba(255,255,255,.18); margin-bottom:16px;
}
.badge{
  display:inline-block; padding:2px 10px; border-radius:999px;
  background:rgba(0,180,216,.12); color:#0B1E3D; font-weight:700;
  margin-right:8px; font-size:.85rem;
}
.ref{ color:#64748b; font-size:.9rem; }
.small-note{ color:var(--muted); font-size:.95rem; }

/* Beautiful Login Box */
.login-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding-top: 5vh;
}
.login-box {
    background: rgba(255, 255, 255, 0.95);
    padding: 50px 40px; 
    border-radius: 24px;
    box-shadow: 0 15px 35px rgba(11, 30, 61, 0.1), 0 5px 15px rgba(0, 0, 0, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    text-align: center;
    width: 100%;
}
.login-title {
    font-size: 26px;
    font-weight: 800;
    color: var(--primary);
    margin-top: 15px;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}
.login-subtitle {
    font-size: 15px;
    color: #64748b;
    margin-bottom: 30px;
}

.stButton>button{
  background:linear-gradient(135deg,var(--primary) 0%,var(--primary2) 100%);
  border:none; border-radius:12px; color:#fff !important;
  padding:.75rem 1.2rem; font-weight:700; width:100%;
}
.stButton>button:hover{
  background:linear-gradient(135deg,var(--primary2) 0%,var(--accent) 100%);
  box-shadow:0 8px 20px rgba(0,180,216,.25);
  transform:translateY(-2px);
}
div[data-baseweb="input"] {
    border-radius: 12px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 🔐 AUTH
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def login_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown('<div class="login-container"><div class="login-box">', unsafe_allow_html=True)
        
        if os.path.exists(APP_CONFIG["logo_path"]):
            st.image(APP_CONFIG["logo_path"], width=100)
        else:
            st.markdown("🛡️", unsafe_allow_html=True)

        st.markdown(f'<div class="login-title">{APP_CONFIG["app_name"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Portal Autentikasi Internal Poltek SSN</div>', unsafe_allow_html=True)

        user = st.text_input("Username", key="login_u", placeholder="Masukkan username Anda")
        pwd = st.text_input("Password", type="password", key="login_p", placeholder="••••••••")
        
        st.write("") 
        if st.button("Masuk Sistem", use_container_width=True, type="primary"):
            if user in USERS and USERS[user]["password"] == pwd:
                st.session_state.authenticated = True
                st.session_state.username = user
                st.session_state.user_role = USERS[user]["role"]
                st.success("Akses Diterima! Mengalihkan...")
                time.sleep(0.8)
                st.rerun()
            else:
                st.error("Kredensial tidak valid. Silakan coba lagi.")
                
        st.markdown("</div></div>", unsafe_allow_html=True)

# =========================================================
# 🧠 EMBEDDINGS / VECTOR / DOCSTORE
# =========================================================
@st.cache_resource
def get_embedding_model():
    if platform.system() == "Windows":
        poppler_path = os.path.join(os.getcwd(), "poppler-lib", "poppler-24.08.0", "Library", "bin")
        tesseract_path = r"C:\Program Files\Tesseract-OCR"
        cur = os.environ.get("PATH", "")
        if poppler_path not in cur and os.path.exists(poppler_path):
            os.environ["PATH"] += os.pathsep + poppler_path
        if tesseract_path not in cur and os.path.exists(tesseract_path):
            os.environ["PATH"] += os.pathsep + tesseract_path

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
        except Exception:
            pass
    return store

def save_docstore(store: InMemoryStore):
    with open(APP_CONFIG["docstore_file"], "wb") as f:
        pickle.dump(store.store, f)

# =========================================================
# 🧹 CLEAN TEXT
# =========================================================
NOISE_PATTERNS = [
    r"\bBADAN SIBER DAN SANDI NEGARA\b",
    r"\bJalan\b", r"\bTelepon\b", r"\bFaksimile\b", r"\bLaman\b", r"\bPos-el\b",
    r"\bTembusan\b", r"\bDokumen ini telah ditandatangani\b", r"\bBSrE\b",
]

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def clean_text(raw: str) -> str:
    if not raw:
        return ""
    t = raw
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\d+\s*[-–]?\s*", s):
            continue
        if any(re.search(p, s, flags=re.IGNORECASE) for p in NOISE_PATTERNS):
            continue
        lines.append(s)

    return "\n".join(lines).strip()

def ref_line(ref: str) -> str:
    return f"<div class='ref'>Rujukan: <b>{ref}</b>.</div>"

# =========================================================
# 📄 PDF PROCESS
# =========================================================
def process_pdf_advanced(file_path: str):
    try:
        return partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res",
            chunking_strategy="by_title",
            max_characters=2200,
            combine_text_under_n_chars=550,
            new_after_n_chars=1600,
        )
    except Exception as e:
        st.warning(f"⚠️ Hi-Res gagal ({e}). Fallback mode cepat...")
        return partition_pdf(filename=file_path, strategy="fast", chunking_strategy="by_title")

# =========================================================
# 📚 PASAL INDEX BUILD/LOAD
# =========================================================
def split_pasal_blocks(text: str) -> List[Tuple[int, str]]:
    if not text:
        return []
    parts = re.split(r"(?=\bPasal\s+\d+)", text, flags=re.IGNORECASE)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"\bPasal\s+(\d+)\b", p, flags=re.IGNORECASE)
        if m:
            out.append((int(m.group(1)), p))
    return out

def extract_ayat(block: str) -> Dict[str, str]:
    ayat: Dict[str, str] = {}
    if not block:
        return ayat
    b = re.sub(r"\bAyat\s*\((\d+)\)", r"(\1)", block, flags=re.IGNORECASE)
    segs = re.split(r"(?=\(\d+\))", b)
    for seg in segs:
        seg = seg.strip()
        m = re.match(r"\((\d+)\)\s*", seg)
        if m:
            ayat[m.group(1)] = seg
    return ayat

def build_pasal_index_from_texts(texts: List[str]) -> Dict[str, Any]:
    merged = "\n\n".join([clean_text(str(t)) for t in texts if t and str(t).strip()])
    merged = re.sub(r"\n{3,}", "\n\n", merged).strip()

    pasal_blocks = split_pasal_blocks(merged)
    idx: Dict[str, Any] = {"pasal": {}, "meta": {"built_at": time.strftime("%Y-%m-%d %H:%M:%S")}}

    for num, block in pasal_blocks:
        key = str(num)
        block_clean = clean_text(block)
        if not block_clean:
            continue
        ay = extract_ayat(block_clean)
        prev = (idx["pasal"].get(key) or {}).get("text", "")
        if len(block_clean) > len(prev):
            idx["pasal"][key] = {"text": block_clean, "ayat": ay}

    return idx

def load_pasal_index() -> Dict[str, Any]:
    if os.path.exists(APP_CONFIG["pasal_index_file"]):
        try:
            with open(APP_CONFIG["pasal_index_file"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"pasal": {}, "meta": {}}
    return {"pasal": {}, "meta": {}}

def save_pasal_index(index: Dict[str, Any]):
    with open(APP_CONFIG["pasal_index_file"], "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def ensure_pasal_index_ready():
    idx = load_pasal_index()
    if idx.get("pasal"):
        return
    store = get_docstore()
    if not store.store:
        return
    texts = [v for _, v in store.store.items()]
    save_pasal_index(build_pasal_index_from_texts(texts))

def list_pasal_numbers(index: Dict[str, Any]) -> List[int]:
    nums = []
    for k in (index.get("pasal") or {}).keys():
        try:
            nums.append(int(k))
        except:
            pass
    return sorted(nums)

# =========================================================
# ✅ CORE PARSERS (PASAL/AYAT/HURUF + LIST)
# =========================================================
def parse_pasal_query(q: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    ql = q.lower()
    m = re.search(r"\bpasal\s+(\d+)\b", ql)
    if not m:
        return None, None, None
    pasal = int(m.group(1))
    m2 = re.search(r"\bayat\s+(\d+)\b", ql)
    ayat = m2.group(1) if m2 else None
    m3 = re.search(r"\bhuruf\s+([a-z])\b", ql)
    huruf = m3.group(1) if m3 else None
    return pasal, ayat, huruf

def extract_section(text: str, start_pat: str, end_pat: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = re.sub(r"-\s*\d+\s*-", " ", text)
    t = re.sub(r"\s+", " ", t).strip()

    m = re.search(start_pat, t, flags=re.IGNORECASE)
    if not m:
        return None
    start = m.start()

    if end_pat:
        m2 = re.search(end_pat, t[m.end():], flags=re.IGNORECASE)
        if m2:
            end = m.end() + m2.start()
            return t[start:end].strip()

    return t[start:].strip()

def parse_lettered_items(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"-\s*\d+\s*-", " ", text)
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()

    m = re.search(r"meliputi\s*:\s*(.*)$", t, flags=re.IGNORECASE)
    if m:
        t = m.group(1).strip()

    t = re.sub(r";\s*dan\s*", "; ", t, flags=re.IGNORECASE).strip()

    matches = list(re.finditer(r"(?<![a-z0-9])([a-i])\.\s*", t, flags=re.IGNORECASE))
    if not matches:
        parts = [p.strip(" ;:,.") for p in t.split(";") if p.strip()]
        return parts[:20]

    items = []
    for i, mm in enumerate(matches):
        start = mm.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        chunk = t[start:end].strip().strip(" ;:,.")
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            items.append(chunk)
    return items

def wants_level(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["tampilkan lengkap", "teks lengkap", "full", "selengkapnya"]):
        return "full"
    if any(k in ql for k in ["jelaskan", "jabarkan", "rinci", "lebih detail", "urai"]):
        return "medium"
    return "short"

def extract_key_lines(text: str, max_lines: int = 4) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    keep = []
    for ln in lines:
        if re.search(r"\b(dilarang|wajib|diperbolehkan|harus)\b", ln, flags=re.IGNORECASE):
            keep.append(ln)
        if len(keep) >= max_lines:
            break
    if not keep:
        keep = lines[:max_lines]
    return keep

def format_answer_responsive(title: str, items: List[str], ref: str) -> str:
    bullets = "\n".join([f"- {x}" for x in items]) if items else "- (Data rinci tidak terdeteksi)"
    return f"**{title}:**\n\n{bullets}\n\n{ref_line(ref)}"

# =========================================================
# ✅ LOCK PASAL 3 (Etos/Kode/Janji) — PRIORITY
# =========================================================
def try_lock_pasal3(idx: Dict[str, Any], ql: str) -> Optional[str]:
    obj = (idx.get("pasal") or {}).get("3")
    if not obj:
        return None
    p3 = obj.get("text") or ""
    if not p3:
        return None

    is_etos = ("etos" in ql and "sandi" in ql)
    is_kode = ("kode" in ql and "kehormatan" in ql)
    is_janji = ("janji" in ql and "taruna" in ql)

    if is_etos:
        sec = extract_section(p3, r"\(\s*3\s*\)\s*Etos\s+Sandi", r"\(\s*4\s*\)\s*Kode\s+Kehormatan")
        if sec:
            items = parse_lettered_items(sec)
            return format_answer_responsive("Etos Sandi meliputi", items, "Pasal 3 ayat (3)")

    if is_kode:
        sec = extract_section(p3, r"\(\s*4\s*\)\s*Kode\s+Kehormatan", r"\(\s*5\s*\)\s*Janji\s+Taruna")
        if sec:
            items = parse_lettered_items(sec)
            return format_answer_responsive("Kode Kehormatan Taruna meliputi", items, "Pasal 3 ayat (4)")

    if is_janji:
        sec = extract_section(p3, r"\(\s*5\s*\)\s*Janji\s+Taruna", r"\bPasal\s+4\b")
        if sec:
            items = parse_lettered_items(sec)
            return format_answer_responsive("Janji Taruna meliputi", items, "Pasal 3 ayat (5)")

    return None

# =========================================================
# ✅ PASAL LOOKUP (Pasal/Ayat/Huruf) — CLEAN OUTPUT
# =========================================================
def pasal_lookup(idx: Dict[str, Any], pasal: int, ayat: Optional[str], huruf: Optional[str], level: str) -> Optional[str]:
    obj = (idx.get("pasal") or {}).get(str(pasal))
    if not obj:
        return None
    ptext = obj.get("text") or ""
    if not ptext:
        return None

    if ayat:
        aytxt = None
        if ayat.isdigit():
            n = int(ayat)
            aytxt = extract_section(ptext, rf"\(\s*{n}\s*\)", rf"\(\s*{n+1}\s*\)")
            if not aytxt:
                aytxt = extract_section(ptext, rf"\(\s*{n}\s*\)", None)

        if not aytxt:
            aytxt = (obj.get("ayat") or {}).get(str(ayat))

        if aytxt:
            if huruf:
                items = parse_lettered_items(aytxt)
                pos = ord(huruf.lower()) - ord("a")
                if 0 <= pos < len(items):
                    return f"**Isi:**\n- {items[pos]}\n\n{ref_line(f'Pasal {pasal} ayat ({ayat}) huruf {huruf}')}"
                lines = extract_key_lines(clean_text(aytxt), 4)
                return f"**Isi (ringkas):**\n" + "\n".join([f"- {x}" for x in lines]) + f"\n\n{ref_line(f'Pasal {pasal} ayat ({ayat})')}"
            
            if level == "full":
                return f"**Teks lengkap:**\n{clean_text(aytxt)}\n\n{ref_line(f'Pasal {pasal} ayat ({ayat})')}"
            lines = extract_key_lines(clean_text(aytxt), 6 if level == "medium" else 4)
            return f"**Isi (ringkas):**\n" + "\n".join([f"- {x}" for x in lines]) + f"\n\n{ref_line(f'Pasal {pasal} ayat ({ayat})')}"

    if level == "full":
        return f"**Teks lengkap:**\n{clean_text(ptext)}\n\n{ref_line(f'Pasal {pasal}')}"
    lines = extract_key_lines(clean_text(ptext), 7 if level == "medium" else 4)
    return f"**Isi (ringkas):**\n" + "\n".join([f"- {x}" for x in lines]) + f"\n\n{ref_line(f'Pasal {pasal}')}"

# ==========================================
# 🧠 CORPUS + VOCAB + TYPO FIX
# ==========================================
def build_corpus_text(max_docs: Optional[int] = None) -> str:
    vs = get_vectorstore()
    store = get_docstore()
    raw = vs._collection.get(include=["metadatas", "documents"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    texts = []
    for i, (doc_text, meta) in enumerate(zip(docs, metas)):
        if max_docs is not None and i >= max_docs:
            break
        doc_id = (meta or {}).get("doc_id")
        if doc_id and doc_id in store.store:
            t = str(store.store[doc_id])
        else:
            t = str(doc_text)
        t = clean_text(t)
        if t:
            texts.append(t)
    return "\n\n".join(texts)

@st.cache_data(show_spinner=False)
def build_vocab_snapshot() -> List[str]:
    corpus = build_corpus_text()
    tokens = re.findall(r"[a-zA-Z0-9]{4,}", corpus.lower())
    return sorted(set(tokens))

STOPWORDS = {
    "apa","yang","di","ke","dari","dan","atau","itu","ini","tentang","pasal",
    "sebutkan","berikan","jelaskan","jabarkan","tolong","dong","nya","kah",
    "taruna","poltek","politeknik","siber","sandi","negara",
    "aturan","peraturan","ketentuan","bab","bagian","ayat","huruf",
    "nomor","tahun","surat","peringatan","dasar","laman","telepon","faksimile",
}

def correct_keyword_typo(word: str, vocab: List[str], cutoff: float = 0.84) -> str:
    if word in vocab:
        return word
    matches = difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff)
    return matches[0] if matches else word

def refine_user_query(original_query: str) -> str:
    prompt = (
        "Perbaiki typo dan buat versi query yang lebih formal dan ringkas untuk pencarian dokumen.\n"
        "JANGAN menjawab pertanyaannya.\n"
        f"Query Asli: {original_query}\n"
        "Query Perbaikan:"
    )
    try:
        resp = chat(
            model=APP_CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        out = (resp.get("message") or {}).get("content", "").strip()
        out = re.sub(r"^\s*query\s*perbaikan\s*:\s*", "", out, flags=re.IGNORECASE).strip()
        return out if out else original_query
    except Exception:
        return original_query

def extract_query_keywords(query: str, refined_query: Optional[str] = None) -> List[str]:
    src = refined_query if refined_query else query
    q = re.sub(r"[^a-zA-Z0-9\s]", " ", src.lower())
    toks = [t.strip() for t in q.split() if t.strip()]
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 4]

    vocab = build_vocab_snapshot()
    out = []
    seen = set()
    for t in toks:
        c = correct_keyword_typo(t, vocab)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:4]

def search_topic_in_pasal_index(index: Dict[str, Any], keywords: List[str]) -> List[Tuple[int, str]]:
    hits = []
    pasal_map = index.get("pasal") or {}
    for k, obj in pasal_map.items():
        try:
            pnum = int(k)
        except:
            continue
        text = (obj.get("text") or "")
        tl = text.lower()
        if any(kw.lower() in tl for kw in keywords):
            hits.append((pnum, text))
    return hits

def format_topic_answer(title: str, pasal_hits: List[Tuple[int, str]]) -> str:
    if not pasal_hits:
        return "Data tidak ditemukan pada dokumen yang dilatih."
    seen = set()
    uniq = []
    for p, t in sorted(pasal_hits, key=lambda x: x[0]):
        if p in seen:
            continue
        seen.add(p)
        uniq.append((p, t))
        if len(uniq) >= 5: 
            break

    out = [f"**Aturan terkait {title} (ringkas):**\n"]
    for p, t in uniq:
        short_text = t[:200].strip() + ("..." if len(t) > 200 else "")
        out.append(f"- **Pasal {p}:** {short_text}")
        
    out.append("\n*Ketik 'Pasal X' untuk melihat isi lengkap peraturannya.*")
    return "\n".join(out)

def keyword_fallback_compact(refined_query: str) -> Optional[str]:
    idx = load_pasal_index()
    kws = extract_query_keywords(refined_query, refined_query=refined_query)
    if not kws:
        return None

    hits = search_topic_in_pasal_index(idx, kws)
    if hits:
        return format_topic_answer(" / ".join(kws), hits)

    corpus = build_corpus_text()
    if not corpus.strip():
        return None

    snippets = []
    for kw in kws:
        low = corpus.lower()
        pos = low.find(kw.lower())
        if pos != -1:
            a = max(0, pos - 1200)
            b = min(len(corpus), pos + 1200)
            snippets.append(corpus[a:b])

    if not snippets:
        return None

    raw = "\n\n".join(snippets)
    blocks = split_pasal_blocks(raw)
    if blocks:
        hits2 = [(p, t) for p, t in blocks]
        return format_topic_answer(" / ".join(kws), hits2)

    short = normalize_whitespace(raw)[:700]
    return f"Berikut kutipan yang ditemukan (ringkas):\n\n- {short}..."

# =========================================================
# 🧾 LOGGING
# =========================================================
def ensure_logs():
    os.makedirs(APP_CONFIG["logs_dir"], exist_ok=True)
    if not os.path.exists(APP_CONFIG["query_log_file"]):
        with open(APP_CONFIG["query_log_file"], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "user", "role", "query", "route", "found", "latency_ms"])

def log_query(user: str, role: str, query: str, route: str, found: str, latency_ms: int):
    ensure_logs()
    with open(APP_CONFIG["query_log_file"], "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user, role, query, route, found, latency_ms])

# =========================================================
# 🧠 SMART ANSWER (NO SPAM, VERY CLEAN)
# =========================================================
def smart_answer(query: str) -> Tuple[str, str]:
    t0 = time.time()
    refined = refine_user_query(query)
    ql = refined.lower()
    idx = load_pasal_index()
    level = wants_level(refined)

    # 0) LOCK PASAL 3 first
    locked = try_lock_pasal3(idx, ql)
    if locked:
        return locked, "lock_pasal3"

    # 1) Pasal lookup
    pasal, ayat, huruf = parse_pasal_query(ql)
    if pasal:
        out = pasal_lookup(idx, pasal, ayat, huruf, level)
        if out:
            return out, "pasal_lookup"

    # 2) TOPIC ROUTES
    topic_map = [
        ("pornografi / kecabulan / erotika", ["pornografi", "kecabulan", "erotika", "eksploitasi seksual", "asusila"]),
        ("internet / media sosial", ["internet", "media sosial", "media elektronik"]),
        ("berbohong", ["berbohong", "tidak didukung fakta", "tidak didukung data"]),
        ("mencuri / pencurian", ["pencurian", "mencuri", "mengambil tanpa hak", "penggelapan"]),
        ("kode kehormatan", ["kode kehormatan", "kehormatan"]),
        ("janji taruna / ikrar", ["janji taruna", "ikrar", "janji"]),
    ]

    for title, keys in topic_map:
        if any(k in ql for k in keys):
            hits = search_topic_in_pasal_index(idx, keys)
            if hits:
                return format_topic_answer(title, hits), f"topic:{title}"

    # 3) SHORT QUERY FALLBACK
    if len(ql) <= 70:
        fb = keyword_fallback_compact(refined)
        if fb:
            return fb, "keyword_fallback"

    # 4) “Daftar pasal”
    if any(k in ql for k in ["daftar pasal", "pasal apa saja", "daftar peraturan", "aturan apa saja"]):
        nums = list_pasal_numbers(idx)
        if not nums:
            return "Index pasal kosong. Latih dokumen dulu.", "catalog_empty"
        show = nums[:80]
        bullets = "\n".join([f"- Pasal {n}" for n in show])
        more = "" if len(nums) <= 80 else f"\n\n(… dan {len(nums)-80} pasal lainnya)"
        return f"**Daftar pasal yang tersedia:**\n\n{bullets}{more}", "catalog"

    # 5) SEMANTIC RAG FALLBACK
    vectorstore = get_vectorstore()
    results_with_score = vectorstore.similarity_search_with_score(refined, k=4)
    docs = [d for d, _ in results_with_score]
    
    if not docs:
        return "Data tidak ditemukan.", "rag_empty"

    store = get_docstore()
    clean_texts = []
    for d in docs:
        doc_id = (d.metadata or {}).get("doc_id")
        if doc_id and doc_id in store.store:
            clean_texts.append(clean_text(store.store[doc_id]))
        else:
            clean_texts.append(clean_text(d.page_content or ""))

    context = "\n\n---\n\n".join([t for t in clean_texts if t])
    if not context.strip():
        return "Data tidak ditemukan.", "rag_empty_context"

    system_prompt = (
        "Anda adalah Asisten Ekstraktor Teks Poltek SSN. "
        "Tugas Anda HANYA menyalin ulang informasi dari DATA PENDUKUNG yang relevan dengan pertanyaan.\n\n"
        "ATURAN MUTLAK:\n"
        "1. DILARANG MERANGKUM ATAU MENJELASKAN. Cukup kutip teks aslinya.\n"
        "2. Jika jawaban tidak ada di DATA PENDUKUNG, Anda WAJIB menjawab persis: 'Maaf, data tidak ditemukan di peraturan.'\n"
        "3. Jangan pernah menggunakan pengetahuan Anda sendiri di luar teks yang diberikan."
    )
    
    user_prompt = f"DATA PENDUKUNG:\n{context}\n\nPERTANYAAN:\n{query}"

    try:
        resp = chat(
            model=APP_CONFIG["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0, "top_p": 0.1},
        )
        ans = (resp.get("message") or {}).get("content", "").strip()
        return (ans if ans else "Data tidak ditemukan."), "rag"
    except Exception as e:
        return f"Terjadi kesalahan pada AI: {e}", "rag_error"

# =========================================================
# 🛠️ ADMIN — Save / Rebuild / CRUD
# =========================================================
def admin_save_data(display_texts: List[str], keywords: List[str]):
    vectorstore = get_vectorstore()
    store = get_docstore()

    doc_ids = [str(uuid.uuid4()) for _ in display_texts]
    docs: List[Document] = []

    for txt, kw, did in zip(display_texts, keywords, doc_ids):
        tx = clean_text(txt or "")
        if not tx:
            continue
        kw = (kw or "").strip()
        content_for_search = f"{tx}\n\n[Meta-Tag/Context: {kw}]" if kw else tx
        docs.append(Document(page_content=content_for_search, metadata={"doc_id": did}))
        store.mset([(did, tx)])

    if not docs:
        st.error("Tidak ada teks valid untuk disimpan.")
        return

    vectorstore.add_documents(docs)
    try:
        vectorstore.persist()
    except Exception:
        pass

    save_docstore(store)

    all_texts = [v for _, v in store.store.items()]
    save_pasal_index(build_pasal_index_from_texts(all_texts))

def rebuild_vectorstore_from_docstore() -> int:
    embeddings = get_embedding_model()
    os.makedirs(APP_CONFIG["db_dir"], exist_ok=True)
    client = chromadb.PersistentClient(path=APP_CONFIG["db_dir"])
    try:
        client.delete_collection(APP_CONFIG["collection_name"])
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=APP_CONFIG["collection_name"],
        embedding_function=embeddings,
        client=client,
    )

    store = get_docstore()
    items = list(store.store.items())

    docs: List[Document] = []
    for doc_id, txt in items:
        tx = clean_text(str(txt))
        if tx:
            docs.append(Document(page_content=tx, metadata={"doc_id": doc_id}))

    if docs:
        vectorstore.add_documents(docs)
    try:
        vectorstore.persist()
    except Exception:
        pass

    save_pasal_index(build_pasal_index_from_texts([v for _, v in store.store.items()]))
    return len(docs)

def page_admin_dashboard():
    st.markdown('<h1 class="main-header">🛠️ Admin Panel</h1>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Latih", "🗄️ Docstore Manager", "🩺 Diagnosa & Rebuild"])

    # ---- TAB 1
    with tab1:
        st.info("Upload PDF → edit teks → tambah keyword → simpan (otomatis update index pasal).")
        os.makedirs(APP_CONFIG["upload_dir"], exist_ok=True)

        if st.session_state.admin_step == 1:
            uploaded_file = st.file_uploader("Pilih PDF", type="pdf")
            if uploaded_file and st.button("🚀 Proses PDF"):
                file_path = os.path.join(APP_CONFIG["upload_dir"], uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.status("Memproses dokumen...", expanded=True) as status:
                    chunks = process_pdf_advanced(file_path)
                    texts = []
                    for c in chunks:
                        if hasattr(c, "text") and c.text:
                            tx = clean_text(c.text)
                            if tx:
                                texts.append(tx)
                    st.session_state.texts = texts
                    st.session_state.keywords = [""] * len(texts)
                    st.session_state.admin_step = 2
                    status.update(label="Selesai. Lanjut ke editor.", state="complete", expanded=False)
                st.rerun()

        elif st.session_state.admin_step == 2:
            st.markdown(f"### Editor ({len(st.session_state.texts)} bagian)")
            c1, c2 = st.columns([1, 5])
            with c1:
                if st.button("⬅️ Kembali"):
                    st.session_state.admin_step = 1
                    st.rerun()

            with st.form("editor_form"):
                for i, tx in enumerate(st.session_state.texts):
                    with st.expander(f"Bagian #{i+1}", expanded=(i < 3)):
                        st.checkbox("Simpan bagian ini", value=True, key=f"keep_{i}")
                        st.text_area("Teks", value=tx, height=140, key=f"txt_{i}")
                        st.text_input("Keyword (opsional)", value=st.session_state.keywords[i], key=f"kw_{i}")

                if st.form_submit_button("💾 Simpan ke Knowledge Base", type="primary"):
                    final_texts, final_kws = [], []
                    for i in range(len(st.session_state.texts)):
                        if st.session_state.get(f"keep_{i}"):
                            final_texts.append(st.session_state.get(f"txt_{i}") or "")
                            final_kws.append(st.session_state.get(f"kw_{i}") or "")
                    with st.spinner("Menyimpan & membangun index..."):
                        admin_save_data(final_texts, final_kws)
                    st.success("Selesai disimpan + index pasal terupdate.")
                    time.sleep(0.6)
                    st.session_state.admin_step = 1
                    st.rerun()

    # ---- TAB 2
    with tab2:
        st.markdown("### Docstore (CRUD sederhana)")
        store = get_docstore()
        all_docs = list(store.store.items())

        if not all_docs:
            st.warning("Docstore kosong.")
        else:
            st.write(f"Total dokumen: {len(all_docs)}")
            doc_to_delete = st.selectbox("Pilih doc_id untuk preview/hapus", [d[0] for d in all_docs])
            preview = str(store.store.get(doc_to_delete, ""))[:700]
            st.info(preview + ("..." if len(preview) >= 700 else ""))

            if st.button("🗑️ Hapus doc ini"):
                try:
                    del store.store[doc_to_delete]
                    save_docstore(store)
                    save_pasal_index(build_pasal_index_from_texts([v for _, v in store.store.items()]))
                    st.success("Dihapus dari docstore + index diperbarui.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal hapus: {e}")

    # ---- TAB 3
    with tab3:
        st.markdown("### Diagnosa Sistem")
        c1, c2, c3 = st.columns(3)
        c1.metric("OS", platform.system())
        c2.metric("Python", platform.python_version())
        c3.metric("Collection", APP_CONFIG["collection_name"])

        st.divider()
        try:
            ollama.list()
            st.success("✅ Ollama ONLINE")
        except Exception:
            st.error("❌ Ollama OFFLINE (jalankan Ollama dulu)")

        idx = load_pasal_index()
        st.info(f"📌 Index pasal: {len(list_pasal_numbers(idx))} pasal")

        try:
            vs = get_vectorstore()
            st.info(f"📌 Vector count: {vs._collection.count()}")
        except Exception as e:
            st.warning(f"Vector count gagal: {e}")

        st.divider()
        if st.button("🧱 Rebuild Vector DB + Index dari Docstore"):
            with st.spinner("Rebuild..."):
                n = rebuild_vectorstore_from_docstore()
            st.success(f"Rebuild selesai. Dokumen dimasukkan: {n}")
            st.rerun()

# =========================================================
# 💬 CHAT
# =========================================================
def page_chat_interface():
    st.markdown('<h1 class="main-header">💬 Konsultasi Pengasuhan</h1>', unsafe_allow_html=True)
    st.markdown(
        "<div class='small-note'>Jawaban dibuat <b>rapi & responsif</b> (isi dulu, rujukan di bawah). "
        "Gunakan: <b>Pasal X ayat Y huruf Z</b> untuk presisi.</div>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("Contoh: 'Sebutkan etos sandi' / 'Sebutkan kode kehormatan' / 'Pasal 3 ayat 1 huruf a'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari..."):
                t0 = time.time()
                ans, route = smart_answer(prompt)
                ms = int((time.time() - t0) * 1000)

                st.markdown(ans, unsafe_allow_html=True)

                try:
                    log_query(
                        user=str(st.session_state.username),
                        role=str(st.session_state.user_role),
                        query=prompt,
                        route=route,
                        found="yes" if "tidak" not in ans.lower() else "maybe",
                        latency_ms=ms,
                    )
                except Exception:
                    pass

                if st.session_state.get("debug_mode"):
                    st.caption(f"route={route} | {ms}ms")

        st.session_state.messages.append({"role": "assistant", "content": ans})

# =========================================================
# 🏠 DASHBOARD
# =========================================================
def page_dashboard():
    idx = load_pasal_index()
    pasal_count = len(list_pasal_numbers(idx))
    st.markdown('<h1 class="main-header">Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="info-card">
  <h3>👋 Halo, {st.session_state.username}!</h3>
  <p>Index pasal terdeteksi: <b>{pasal_count}</b></p>
  <p>Contoh pertanyaan:</p>
  <ul>
    <li><b>Sebutkan etos sandi</b></li>
    <li><b>Sebutkan kode kehormatan</b></li>
    <li><b>Sebutkan janji taruna</b></li>
    <li><b>Pasal 3 ayat 1 huruf a</b></li>
    <li><b>Daftar pasal</b></li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================================================
# 🚀 MAIN
# =========================================================
def main():
    init_session_state()

    if not st.session_state.authenticated:
        login_page()
        return

    ensure_pasal_index_ready()

    with st.sidebar:
        if os.path.exists(APP_CONFIG["logo_path"]):
            st.image(APP_CONFIG["logo_path"], width=160)
        else:
             st.markdown("🛡️", unsafe_allow_html=True)

        st.markdown(f"### {APP_CONFIG['app_name']}")
        st.markdown(
            f"""
<div class="user-badge">
  <div style="font-weight:800; font-size:1.05rem;">👤 {st.session_state.username}</div>
  <div style="opacity:.85; font-size:.9rem;">{st.session_state.user_role}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        menu = ["Dashboard", "Konsultasi AI"]
        if st.session_state.user_role == "Super Admin":
            menu.insert(1, "Admin Panel")

        selected = st.radio("Navigasi", menu, label_visibility="collapsed")
        st.divider()

        if st.session_state.user_role == "Super Admin":
            st.session_state.debug_mode = st.toggle("🔧 Debug Mode", value=st.session_state.debug_mode)

        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()

        st.caption(f"Ver: {APP_CONFIG['version']}")

    if selected == "Dashboard":
        page_dashboard()
    elif selected == "Admin Panel":
        page_admin_dashboard()
    else:
        page_chat_interface()

if __name__ == "__main__":
    main()