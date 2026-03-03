import streamlit as st
import os
import uuid
import base64
import ollama
import pickle
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from ollama import chat

# Fungsi untuk memproses PDF
def process_pdf(file_path):
    print("Processing PDF...")
    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=400,
        new_after_n_chars=1200,
    )
    return chunks

# Fungsi untuk mengekstrak teks
def extract_texts(chunks):
    print("Extracting Texts...")
    texts = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            chunk_text = []
            for el in chunk_els:
                if 'Image' not in str(type(el)):
                    chunk_text.append(el.text)
            if chunk_text:
                texts.append(" ".join(chunk_text))
    return texts

# Fungsi untuk menampilkan dan mengumpulkan detail gambar
def display_and_input_image_details(chunks):
    print("Displaying Images and Inputting Details...")
    image_names = []
    image_descriptions = []
    image_data_list = []
    image_skipped = []
    
    for i, chunk in enumerate(chunks):
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for j, el in enumerate(chunk_els):
                if "Image" in str(type(el)):
                    img_base64 = el.metadata.image_base64
                    img_data = base64.b64decode(img_base64)
                    st.image(img_data, caption=f"Gambar {i+1}-{j+1}", use_container_width=True)
                    
                    # Checkbox untuk melewati gambar
                    skip_image = st.checkbox(f"Lewati gambar {i+1}-{j+1} (misalnya, logo atau dekoratif)", key=f"skip_{i}_{j}")
                    
                    if not skip_image:
                        name = st.text_input(f"Masukkan nama gambar {i+1}-{j+1}:", key=f"name_{i}_{j}")
                        description = st.text_input(f"Masukkan deskripsi gambar {i+1}-{j+1}:", key=f"desc_{i}_{j}")
                        if name and description:
                            image_names.append(name)
                            image_descriptions.append(description)
                            image_data_list.append(img_data)
                            image_skipped.append(False)
                            st.info(f"Data untuk gambar {i+1}-{j+1} telah dikumpulkan. Akan disimpan saat tombol SUBMIT ditekan.")
                        else:
                            st.warning(f"Harap isi nama dan deskripsi untuk gambar {i+1}-{j+1}, atau centang 'Lewati' untuk mengabaikan.")
                            image_names.append(None)
                            image_descriptions.append(None)
                            image_data_list.append(img_data)
                            image_skipped.append(False)
                    else:
                        st.info(f"Gambar {i+1}-{j+1} dilewati dan tidak akan diproses.")
                        print(f"Gambar {i+1}-{j+1} dilewati oleh admin.")
                        image_names.append(None)
                        image_descriptions.append(None)
                        image_data_list.append(img_data)
                        image_skipped.append(True)
    
    return image_names, image_descriptions, image_data_list, image_skipped

# Fungsi untuk meringkas teks
def summarize_texts(texts):
    print("Summarizing Texts...")
    text_summarization = []
    messages = [
        {
            "role": "system",
            "content": "Anda adalah seorang asisten yang ditugaskan untuk menjelaskan ulang teks yang diberikan kepada Anda. Teks tersebut adalah teks dengan konteks sertifikasi keamanan produk teknologi informasi. Lakukan tugas Anda tanpa komentar atau  tanpa asumsi tambahan apapun. Berikan ringkasan apa adanya berupa paragraf.",
        },
        {
            "role": "user",
            "content": "[PELAYANAN SERTIFIKASI CC INDONESIA - TARGET OF EVALUATION (TOE) – Tahap I] Dokumen Persyaratan Sertifikasi: ✓ Formulir Aplikasi Permohonan Sertifikasi ✓ Security Target (ST) ✓ Evaluation Project Proposal (EPP) ✓ Surat pernyataan dokumen penilaian awal ST dan kecukupan bukti evaluasi dari Laboratorium Pengujian yang berlisensi ✓ Target of Evaluation (TOE) Note: ✓ Bukti evaluasi* : Security Target (ST), guidance documentation, licecycle documentation, architecture and design documentation, testing documentation ✓ Agenda rapat pembuka: Penandatanganan Perjanjian Sertifikasi, Non-Disclosure Agreement (NDA), Kontrak Pengujian ✓ * bukti evaluasi berupa ST saja sudah dapat diterima permohonan sertifikasinya",
        },
        {
            "role": "assistant",
            "content": "Dokumen persyaratan untuk sertifikasi keamanan produk teknologi informasi mencakup beberapa dokumen penting, yaitu:\n1. Formulir Aplikasi Permohonan Sertifikasi\n2. Security Target (ST)\n3. Evaluation Project Proposal (EPP)\n4. Surat pernyataan dokumen penilaian awal ST dan kecukupan bukti evaluasi dari Laboratorium Pengujian yang berlisensi\n\nUntuk dapat menerima permohonan sertifikasinya, harus ada bukti evaluasi berupa:\n- Security Target (ST)\n- Dokumentasi pengembangan arsitekturnya\n- Dokumentasi perizinan atau lisensinya\n- Dokumentasi tes dan penilaian\n\nAgenda rapat pembuka mencakup beberapa dokumen penting, yaitu:\n1. Perjanjian Sertifikasi\n2. Non-Disclosure Agreement (NDA)\n3. Kontrak Pengujian",
        },
    ]

    for text in texts:
        # user_input = text
        # messages.append({'role': 'user', 'content': user_input})  # Tambahkan riwayat input user
        # response = ollama.chat(
        #     model='qwen3:1.7b',
        #     messages=messages[-4:],  # Ambil hanya 4 riwayat terbaru
        #     stream=False,
        # )
        # output = response['message']['content']
        # messages.append({'role': 'assistant', 'content': output})  # Tambahkan jawaban dari assistant
        text_summarization.append(text)
    return text_summarization

# Fungsi untuk menyimpan dokumen
def save_document(texts, text_summarization, image_names, image_descriptions):
    print("Saving documents for future retrieval...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings, persist_directory="./db")
    store = InMemoryStore()

    # Load existing documents if available
    if os.path.exists("./docstore.pkl"):
        with open("./docstore.pkl", "rb") as f:
            existing_docstore = pickle.load(f)
        store.mset(list(existing_docstore.items()))

    id_key = "doc_id"

    # Generate document IDs for texts and summaries
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_docs = [Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summarization)]
    
    # Add text summaries to vectorstore
    vectorstore.add_documents(summary_docs)
    store.mset(list(zip(doc_ids, texts)))

    # Only process image-related documents if there are images
    if image_names and image_descriptions:  # Check if both lists are non-empty
        img_ids = [str(uuid.uuid4()) for _ in image_names]
        image_docs = [Document(page_content=desc, metadata={id_key: img_ids[i]}) for i, desc in enumerate(image_descriptions) if desc]
        
        # Add image descriptions to vectorstore
        vectorstore.add_documents(image_docs)
        
        # Store image paths in the docstore
        image_paths = [f"./images/{name}.jpg" for name in image_names]
        store.mset(list(zip(img_ids, image_paths)))

    # Load existing texts and save new ones
    existing_texts = []
    if os.path.exists("./texts.txt"):
        with open("./texts.txt", "r", encoding="utf-8") as f:
            existing_texts = f.read().splitlines()

    all_texts = existing_texts + texts
    with open("./texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_texts))

    # Save the docstore
    with open("./docstore.pkl", "wb") as f:
        pickle.dump(store.store, f)

    # Persist the vectorstore
    vectorstore.persist()

    print("Data saved successfully for future retrieval.")

# Fungsi utama untuk admin
def main():
    st.title("Document Ingestion Interface")
    
    upload_dir = "./uploaded/"
    image_dir = "./images/"
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "texts" not in st.session_state:
        st.session_state.texts = None
    if "image_names" not in st.session_state:
        st.session_state.image_names = []
    if "image_descriptions" not in st.session_state:
        st.session_state.image_descriptions = []
    if "image_data_list" not in st.session_state:
        st.session_state.image_data_list = []
    if "image_skipped" not in st.session_state:
        st.session_state.image_skipped = []
    if "text_summarization" not in st.session_state:
        st.session_state.text_summarization = []
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "review_mode" not in st.session_state:
        st.session_state.review_mode = False
    
    if not st.session_state.review_mode:
        uploaded_file = st.file_uploader("Unggah PDF", type="pdf")
        
        if uploaded_file is not None:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if os.path.exists(file_path):
                if st.session_state.chunks is None:
                    st.session_state.chunks = process_pdf(file_path)
                
                if st.session_state.texts is None:
                    st.session_state.texts = extract_texts(st.session_state.chunks)
                
                if not st.session_state.submitted:
                    st.session_state.image_names, st.session_state.image_descriptions, st.session_state.image_data_list, st.session_state.image_skipped = display_and_input_image_details(st.session_state.chunks)
                    
                    if st.button("SUBMIT"):
                        # Validasi input gambar
                        valid = True
                        seen_names = set()
                        for idx, (name, desc, skipped) in enumerate(zip(st.session_state.image_names, st.session_state.image_descriptions, st.session_state.image_skipped)):
                            if not skipped:
                                if not name or not desc:
                                    st.error(f"Gambar {idx+1} tidak dilewati tetapi nama atau deskripsi kosong. Harap lengkapi atau lewati gambar.")
                                    valid = False
                                elif name in seen_names:
                                    st.error(f"Nama gambar '{name}' duplikat. Harap gunakan nama unik.")
                                    valid = False
                                elif os.path.exists(f"./images/{name}.jpg"):
                                    st.error(f"Nama gambar '{name}' sudah ada di direktori ./images/. Harap gunakan nama lain.")
                                    valid = False
                                else:
                                    seen_names.add(name)
                        
                        if valid:
                            # Simpan semua gambar ke disk
                            for name, img_data, skipped in zip(st.session_state.image_names, st.session_state.image_data_list, st.session_state.image_skipped):
                                if not skipped and name:
                                    image_path = f"./images/{name}.jpg"
                                    with open(image_path, "wb") as f:
                                        f.write(img_data)
                                    print(f"Gambar disimpan dengan nama: {name}")
                            
                            # Lanjutkan dengan ringkasan teks
                            st.session_state.text_summarization = summarize_texts(st.session_state.texts)
                            st.session_state.submitted = True
                            st.session_state.review_mode = True
                            st.rerun()
                        else:
                            st.error("Validasi gagal. Perbaiki input gambar sebelum submit.")
            else:
                st.error(f"Gagal menyimpan file ke {file_path}")
        
        if st.button("Reset"):
            st.session_state.chunks = None
            st.session_state.texts = None
            st.session_state.image_names = []
            st.session_state.image_descriptions = []
            st.session_state.image_data_list = []
            st.session_state.image_skipped = []
            st.session_state.text_summarization = []
            st.session_state.submitted = False
            st.session_state.review_mode = False
            st.success("Sesi telah direset. Silakan unggah PDF baru.")
    
    if st.session_state.review_mode:
        st.subheader("Review Data Sebelum Menyimpan ke Retriever")
        
        st.write("### Daftar Chunk Teks dan Ringkasan")
        if st.session_state.texts and st.session_state.text_summarization:
            for i, (text, summary) in enumerate(zip(st.session_state.texts, st.session_state.text_summarization)):
                with st.expander(f"Chunk Teks {i+1}"):
                    st.write("**Teks Asli:**")
                    st.write(text)
                    st.write("**Ringkasan (Dapat Diedit):**")
                    edited_summary = st.text_area(f"Edit ringkasan untuk chunk {i+1}", value=summary, key=f"summary_{i}")
                    st.session_state.text_summarization[i] = edited_summary
        else:
            st.warning("Tidak ada teks atau ringkasan untuk ditampilkan.")
        
        st.write("### Daftar Gambar")
        if st.session_state.image_names and st.session_state.image_descriptions and st.session_state.image_data_list:
            for i, (name, desc, img_data, skipped) in enumerate(zip(st.session_state.image_names, st.session_state.image_descriptions, st.session_state.image_data_list, st.session_state.image_skipped)):
                if not skipped and name and desc:
                    st.image(img_data, caption=f"Gambar: {name}", use_container_width=True)
                    st.write(f"**Nama Gambar:** {name}")
                    st.write(f"**Deskripsi:** {desc}")
                else:
                    st.info(f"Gambar {i+1} dilewati atau tidak memiliki nama/deskripsi.")
        else:
            st.info("Tidak ada gambar yang diproses (semua gambar mungkin dilewati).")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Simpan ke Knowledge Base"):
                save_document(
                    st.session_state.texts,
                    st.session_state.text_summarization,
                    [name for name, skipped in zip(st.session_state.image_names, st.session_state.image_skipped) if not skipped and name],
                    [desc for desc, skipped in zip(st.session_state.image_descriptions, st.session_state.image_skipped) if not skipped and desc]
                )
                st.success("Knowledge Base telah dibangun dan data disimpan di server!")
                st.session_state.chunks = None
                st.session_state.texts = None
                st.session_state.image_names = []
                st.session_state.image_descriptions = []
                st.session_state.image_data_list = []
                st.session_state.image_skipped = []
                st.session_state.text_summarization = []
                st.session_state.submitted = False
                st.session_state.review_mode = False
                st.rerun()
        with col2:
            if st.button("Kembali ke Input Gambar"):
                st.session_state.submitted = False
                st.session_state.review_mode = False
                st.rerun()

if __name__ == "__main__":
    main()