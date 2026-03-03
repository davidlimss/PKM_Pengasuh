import os
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Memulai AI... Harap tunggu sebentar (inisialisasi model).")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from ollama import chat


# ==== KONFIGURASI MODEL OLLAMA ====
# Ganti ini kalau kamu pakai model lain, misalnya "llama3.2:1b"
OLLAMA_MODEL_NAME = "qwen2:1.5b"


# ==== LOAD VECTORSTORE (DB) YANG DIBUAT DARI admin_app_rev.py ====
embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")

vectorstore = Chroma(
    collection_name="multi_modal_rag",
    embedding_function=embeddings,
    persist_directory="./db",  # folder DB yang tadi dibuat
)


def ask_knowledge_base(question: str) -> str:
    # 1. Cari potongan teks paling relevan dari Chroma
    docs = vectorstore.similarity_search(question, k=6)

    if not docs:
        return "Aku tidak menemukan jawaban di dokumen yang ada."

    # Gabungkan konteks
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # (Opsional) kalau mau lihat konteks yang kepake, buka komentar ini:
    # print("\n--- KONTEKS YANG DIAMBIL ---\n")
    # print(context[:2000])
    # print("\n----------------------------\n")

    # 2. Pesan ke model: sistem prompt dibuat LEBIH KETAT
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah asisten hukum yang HARUS menjawab hanya berdasarkan teks dalam CONTEXT.\n"
                "Aturan ketat:\n"
                "1. Jangan menambah, mengarang, atau menyimpulkan hal yang tidak tertulis di CONTEXT.\n"
                "2. Jika ditanya tentang 'larangan', hanya sebut hal-hal yang secara eksplisit mengandung kata:\n"
                "   - 'dilarang', 'tidak boleh', 'larangan', atau sinonim langsung yang jelas.\n"
                "3. Jika memungkinkan, sebutkan pasal dan ayat sebagaimana tertulis di CONTEXT.\n"
                "4. Jika di CONTEXT tidak ada informasi yang cukup untuk menjawab, jawab dengan jujur:\n"
                "   'Dalam konteks ini tidak ditemukan ketentuan yang secara jelas menjawab pertanyaan.'\n"
                "5. Jangan menganggap logo, tanda tangan, gambar, atau lambang negara sebagai 'larangan'.\n"
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nPERTANYAAN:\n{question}",
        },
    ]

    # 3. Panggil Ollama lokal
    result = chat(
        model=OLLAMA_MODEL_NAME,
        messages=messages,
        # Temperatur 0 supaya nggak ngarang-ngarang
        options={"temperature": 0},
    )

    return result["message"]["content"]


if __name__ == "__main__":
    print("Chat ke knowledge base Perdir kamu. Kosongkan pertanyaan untuk keluar.\n")
    while True:
        try:
            q = input("Tanya apa: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not q:
            break

        answer = ask_knowledge_base(q)
        print("\n=== Jawaban ===")
        print(answer)
        print("==============\n")
