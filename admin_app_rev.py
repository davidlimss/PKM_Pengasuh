import streamlit as st
import os
import uuid
import base64
import ollama
import pickle
from unstructured.partition.pdf import partition_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from ollama import chat

# --- FUNGSI HELPER ---

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

def display_and_input_image_details(chunks):
    image_names = []
    image_descriptions = []
    image_data_list = []
    image_skipped = []
    img_counter = 0

    for i, chunk in enumerate(chunks):
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for j, el in enumerate(chunk_els):
                if "Image" in str(type(el)):
                    img_counter += 1
                    img_base64 = el.metadata.image_base64
                    img_data = base64.b64decode(img_base64)
                    
                    st.markdown(f"### Gambar #{img_counter}")
                    st.image(img_data, width=300)
                    
                    skip_image = st.checkbox(f"Lewati Gambar #{img_counter} (misal: logo/dekoratif)", key=f"skip_{i}_{j}")
                    
                    if not skip_image:
                        name = st.text_input(f"Nama File Gambar #{img_counter}:", key=f"name_{i}_{j}")
                        description = st.text_area(f"Deskripsi Gambar #{img_counter}:", key=f"desc_{i}_{j}")
                        image_names.append(name)
                        image_descriptions.append(description)
                        image_data_list.append(img_data)
                        image_skipped.append(False)
                    else:
                        image_names.append(None)
                        image_descriptions.append(None)
                        image_data_list.append(img_data)
                        image_skipped.append(True)
                    st.divider()
    
    return image_names, image_descriptions, image_data_list, image_skipped

def summarize_texts(texts):
    print("Summarizing Texts...")
    text_summarization = []
    # Menggunakan placeholder ringkasan untuk performa.
    # Jika ingin pakai AI, uncomment bagian ollama di bawah.
    for text in texts:
        # messages = [{'role': 'user', 'content': f"Ringkas teks berikut: {text}"}]
        # response = chat(model='qwen2:1.5b', messages=messages)
        # text_summarization.append(response['message']['content'])
        text_summarization.append(text[:200] + "... (Ringkasan otomatis)")
    return text_summarization

def save_document(texts, text_summarization, image_names, image_descriptions):
    print("Saving documents...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings, persist_directory="./db")
    store = InMemoryStore()

    if os.path.exists("./docstore.pkl"):
        with open("./docstore.pkl", "rb") as f:
            existing_docstore = pickle.load(f)
        store.mset(list(existing_docstore.items()))

    id_key = "doc_id"
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    
    summary_docs = [Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summarization)]
    vectorstore.add_documents(summary_docs)
    store.mset(list(zip(doc_ids, texts)))

    if image_names and image_descriptions:
        img_ids = [str(uuid.uuid4()) for _ in image_names]
        image_docs = [Document(page_content=desc, metadata={id_key: img_ids[i]}) for i, desc in enumerate(image_descriptions) if desc]
        vectorstore.add_documents(image_docs)
        image_paths = [f"./images/{name}.jpg" for name in image_names]
        store.mset(list(zip(img_ids, image_paths)))

    with open("./docstore.pkl", "wb") as f:
        pickle.dump(store.store, f)
    print("Data saved.")

# --- MAIN APPLICATION ---

def main():
    st.set_page_config(layout="wide", page_title="Admin - Ingestion System")
    st.title("📄 Document Ingestion (Sortir & Edit)")
    
    os.makedirs("./uploaded/", exist_ok=True)
    os.makedirs("./images/", exist_ok=True)
    
    # State Initialization
    if "chunks" not in st.session_state: st.session_state.chunks = None
    if "texts" not in st.session_state: st.session_state.texts = None
    if "step" not in st.session_state: st.session_state.step = 1 # Steps: 1=Upload, 2=Sort/Edit, 3=Images, 4=Final
    
    if "image_names" not in st.session_state: st.session_state.image_names = []
    if "image_descriptions" not in st.session_state: st.session_state.image_descriptions = []
    if "image_data_list" not in st.session_state: st.session_state.image_data_list = []
    if "image_skipped" not in st.session_state: st.session_state.image_skipped = []
    if "text_summarization" not in st.session_state: st.session_state.text_summarization = []

    # --- STEP 1: UPLOAD ---
    if st.session_state.step == 1:
        st.info("Langkah 1: Unggah dokumen PDF.")
        uploaded_file = st.file_uploader("Unggah PDF", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Proses Dokumen"):
                file_path = os.path.join("./uploaded/", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Sedang memecah PDF (Chunking)..."):
                    st.session_state.chunks = process_pdf(file_path)
                    st.session_state.texts = extract_texts(st.session_state.chunks)
                    st.session_state.step = 2 
                    st.rerun()

    # --- STEP 2: SORTIR & EDIT TEKS (UPDATE FITUR INI) ---
    elif st.session_state.step == 2:
        st.info("Langkah 2: Sortir Chunk. Centang 'Simpan' untuk memakai chunk, hilangkan centang untuk membuang (misal: header/sampah). Kamu juga bisa mengedit teksnya.")
        
        total_chunks = len(st.session_state.texts)
        st.markdown(f"### Ditemukan {total_chunks} Potongan Teks")
        
        # Container untuk form
        with st.form("sort_edit_form"):
            # Loop untuk menampilkan editor setiap chunk dengan Checkbox
            for i in range(total_chunks):
                with st.expander(f"Chunk #{i+1}", expanded=True):
                    col1, col2 = st.columns([1, 10])
                    
                    with col1:
                        # Checkbox Sortir
                        st.markdown("<br>", unsafe_allow_html=True) # Spacer
                        keep = st.checkbox("Simpan?", value=True, key=f"keep_{i}")
                    
                    with col2:
                        # Editor Teks
                        st.text_area(
                            f"Isi Teks Chunk #{i+1}", 
                            value=st.session_state.texts[i], 
                            height=150,
                            key=f"text_area_{i}",
                            label_visibility="collapsed"
                        )
            
            # Tombol submit form
            submit_sort = st.form_submit_button("✅ Selesai Sortir & Lanjut")
            
            if submit_sort:
                final_texts = []
                discarded_count = 0
                
                # Loop ulang untuk mengambil data yang DICENTANG saja
                for i in range(total_chunks):
                    is_kept = st.session_state.get(f"keep_{i}", True)
                    edited_text = st.session_state.get(f"text_area_{i}", "")
                    
                    if is_kept and edited_text.strip():
                        final_texts.append(edited_text)
                    else:
                        discarded_count += 1
                
                if not final_texts:
                    st.error("Waduh, kamu membuang semua chunk! Harap pilih minimal satu.")
                else:
                    st.session_state.texts = final_texts # Update list teks utama dengan hasil sortir
                    st.success(f"{len(final_texts)} chunk disimpan, {discarded_count} chunk dibuang.")
                    
                    # Lanjut ke langkah berikutnya
                    st.session_state.step = 3
                    st.rerun()

        if st.button("⬅️ Batal/Upload Ulang"):
            st.session_state.step = 1
            st.session_state.chunks = None
            st.rerun()

    # --- STEP 3: GAMBAR ---
    elif st.session_state.step == 3:
        st.info("Langkah 3: Identifikasi Gambar (Optional).")
        
        # Helper function untuk display, logika save ada di tombol bawah
        names, descs, datas, skips = display_and_input_image_details(st.session_state.chunks)
        
        if st.button("Simpan Gambar & Buat Ringkasan"):
            valid = True
            collected_names = []
            collected_descs = []
            collected_data = []
            collected_skips = []
            
            # Logika pengambilan data gambar dari session state
            img_idx = 0
            for i, chunk in enumerate(st.session_state.chunks):
                if "CompositeElement" in str(type(chunk)):
                     chunk_els = chunk.metadata.orig_elements
                     for j, el in enumerate(chunk_els):
                        if "Image" in str(type(el)):
                            skip_key = f"skip_{i}_{j}"
                            name_key = f"name_{i}_{j}"
                            desc_key = f"desc_{i}_{j}"
                            
                            is_skipped = st.session_state.get(skip_key, False)
                            name_val = st.session_state.get(name_key, "")
                            desc_val = st.session_state.get(desc_key, "")
                            
                            img_base64 = el.metadata.image_base64
                            img_data = base64.b64decode(img_base64)

                            if not is_skipped:
                                if not name_val or not desc_val:
                                    st.error(f"Error: Gambar #{img_idx+1} dipilih tapi data kosong.")
                                    valid = False
                                else:
                                    collected_names.append(name_val)
                                    collected_descs.append(desc_val)
                                    collected_data.append(img_data)
                                    collected_skips.append(False)
                            else:
                                collected_names.append(None)
                                collected_descs.append(None)
                                collected_data.append(img_data)
                                collected_skips.append(True)
                            img_idx += 1

            if valid:
                st.session_state.image_names = collected_names
                st.session_state.image_descriptions = collected_descs
                st.session_state.image_data_list = collected_data
                st.session_state.image_skipped = collected_skips
                
                # Save physical images
                for name, data, skipped in zip(collected_names, collected_data, collected_skips):
                    if not skipped and name:
                        with open(f"./images/{name}.jpg", "wb") as f:
                            f.write(data)
                
                with st.spinner("Membuat ringkasan AI..."):
                    st.session_state.text_summarization = summarize_texts(st.session_state.texts)
                
                st.session_state.step = 4
                st.rerun()

    # --- STEP 4: FINAL REVIEW ---
    elif st.session_state.step == 4:
        st.subheader("Langkah 4: Review Akhir & Simpan")
        
        st.write("### 📝 Daftar Teks Final")
        for i, (text, summary) in enumerate(zip(st.session_state.texts, st.session_state.text_summarization)):
            with st.expander(f"Chunk Final #{i+1}"):
                st.info(text)
                st.caption(f"Ringkasan: {summary}")

        col1, col2 = st.columns([1, 4])
        with col1:
             if st.button("⬅️ Kembali"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("💾 SIMPAN KE DATABASE"):
                with st.spinner("Menyimpan..."):
                    save_document(
                        st.session_state.texts,
                        st.session_state.text_summarization,
                        [n for n, s in zip(st.session_state.image_names, st.session_state.image_skipped) if not s],
                        [d for d, s in zip(st.session_state.image_descriptions, st.session_state.image_skipped) if not s]
                    )
                st.success("Selesai! Data tersimpan.")
                if st.button("Mulai Baru"):
                    st.session_state.step = 1
                    st.session_state.chunks = None
                    st.rerun()

if __name__ == "__main__":
    main()