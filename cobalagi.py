import streamlit as st
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import os
import tiktoken

# Import pengecualian spesifik dari OpenAI
from openai import OpenAIError, APIError, AuthenticationError, RateLimitError

# --- Streamlit UI: Sidebar ---
st.sidebar.title("Pengaturan")

# Input API Key di Sidebar
openai_api_key = st.sidebar.text_input(
    "Masukkan OpenAI API Key Anda",
    type="password",
    help="Dapatkan kunci API Anda dari platform.openai.com"
)

# Set OpenAI API Key dan Inisialisasi Klien
client_openai = None # Inisialisasi di luar if agar bisa diakses global
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key # Ini untuk kompatibilitas dengan beberapa fungsi lama jika ada
    st.sidebar.success("OpenAI API Key berhasil diatur!")
    
    try:
        client_openai = openai.OpenAI() # Inisialisasi klien OpenAI baru
    except Exception as e:
        st.sidebar.error(f"Gagal menginisialisasi klien OpenAI: {e}")
        st.stop() # Hentikan jika klien OpenAI tidak bisa diinisialisasi
else:
    st.sidebar.warning("Harap masukkan OpenAI API Key Anda untuk melanjutkan.")
    st.stop() # Menghentikan eksekusi script lebih lanjut jika API key belum diisi

# --- Inisialisasi Model dan ChromaDB (setelah API Key diatur) ---

# Inisialisasi model embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

# Inisialisasi ChromaDB client
@st.cache_resource
def get_chroma_client():
    try:
        # Mencoba membuat koneksi persisten ke ChromaDB
        # Penting: Pastikan folder ./chroma_db memiliki izin tulis
        return chromadb.PersistentClient(path="./chroma_db") 
    except Exception as e:
        st.error(f"Gagal menginisialisasi ChromaDB PersistentClient: {e}")
        st.info("Pastikan Anda memiliki izin tulis di direktori saat ini dan tidak ada proses lain yang mengunci folder 'chroma_db'.")
        st.stop() # Hentikan aplikasi jika ChromaDB tidak bisa diinisialisasi
        
client = get_chroma_client()

# --- Fungsi-fungsi Utama ---

# Fungsi untuk memuat dan membagi teks dari PDF
def load_and_split_pdf(uploaded_file):
    reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    # Simple chunking by paragraph/sentence for now
    chunks = [t.strip() for t in text.split('\n\n') if t.strip()]
    return chunks

# Fungsi untuk menambahkan dokumen ke ChromaDB
def add_documents_to_chroma(collection_name, texts):
    collection = client.get_or_create_collection(name=collection_name)
    
    embeddings = model.encode(texts).tolist()
    
    # Generate unique IDs based on collection name and index
    current_ids_in_collection = collection.get()['ids']
    new_ids = []
    for i, text_chunk in enumerate(texts):
        potential_id = f"{collection_name}_doc_{i}"
        counter = 0
        while potential_id in current_ids_in_collection:
            counter += 1
            potential_id = f"{collection_name}_doc_{i}_{counter}"
        new_ids.append(potential_id)

    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=new_ids
    )
    st.success(f"Berhasil mengunggah {len(texts)} chunks ke koleksi '{collection_name}'")

# Fungsi untuk melakukan pencarian di ChromaDB
def retrieve_documents(query, collection_name, n_results=4):
    try:
        collection = client.get_collection(name=collection_name)
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return results['documents'][0] if results and 'documents' in results and results['documents'] else []
    except Exception as e:
        st.error(f"Error saat mengambil dokumen dari ChromaDB: {e}")
        return []

# Fungsi untuk membuat prompt RAG
def create_rag_prompt(query, context_docs):
    context = "\n\n".join(context_docs)
    prompt = f"""Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan dokumen yang diberikan.
    Jawab pertanyaan pengguna hanya berdasarkan informasi yang ditemukan dalam konteks berikut.
    Jika Anda tidak dapat menemukan jawabannya dalam konteks yang diberikan, katakan saja bahwa Anda tidak tahu.

    Konteks:
    {context}

    Pertanyaan: {query}
    Jawaban:
    """
    return prompt

# Fungsi untuk berinteraksi dengan OpenAI GPT
def generate_response(prompt):
    global client_openai # Pastikan kita mengakses client_openai global
    if not client_openai:
        st.error("OpenAI client belum diinisialisasi. Harap masukkan API Key.")
        return None
    
    try:
        response = client_openai.chat.completions.create( # Menggunakan .chat.completions.create
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Anda adalah asisten yang membantu."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content # Akses konten
    except AuthenticationError:
        st.error("OpenAI Authentication Error: API Key Anda mungkin salah atau kedaluwarsa.")
        return None
    except RateLimitError:
        st.error("OpenAI Rate Limit Error: Anda terlalu sering membuat permintaan, coba lagi nanti.")
        return None
    except APIError as e:
        st.error(f"OpenAI API Error: {e.status_code} - {e.response.json()}") # Tampilkan detail error dari response
        return None
    except OpenAIError as e: # Tangkap kesalahan OpenAI lainnya
        st.error(f"Terjadi kesalahan OpenAI umum: {e}")
        return None
    except Exception as e: # Tangkap kesalahan non-OpenAI lainnya
        st.error(f"Terjadi kesalahan tak terduga saat menghubungi OpenAI: {e}")
        return None

# --- Streamlit UI: Main Content ---
st.title("Chat dengan Dokumen PDF Anda (RAG)")

# Bagian Unggah PDF
st.header("1. Unggah Dokumen PDF Anda")
uploaded_file = st.file_uploader("Pilih file PDF", type="pdf")
new_collection_name = st.text_input("Nama Koleksi Baru untuk Dokumen Ini:", "my_pdf_collection")

if uploaded_file and st.button("Proses PDF dan Tambahkan ke ChromaDB"):
    if not new_collection_name:
        st.error("Nama koleksi tidak boleh kosong.")
    else:
        with st.spinner("Memproses PDF dan membuat embedding..."):
            try:
                chunks = load_and_split_pdf(uploaded_file)
                if chunks:
                    add_documents_to_chroma(new_collection_name, chunks)
                    st.session_state.current_collection = new_collection_name # Set koleksi aktif
                else:
                    st.warning("PDF kosong atau tidak dapat diekstraksi teks.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses PDF: {e}")

st.divider()

# Bagian Daftar Koleksi yang Sudah Ada
st.sidebar.header("Koleksi ChromaDB yang Ada")
try:
    existing_collections = client.list_collections()
    if existing_collections:
        collection_names = [col.name for col in existing_collections]
        st.sidebar.write("Pilih koleksi untuk chatting:")
        
        # Buat dropdown untuk memilih koleksi
        selected_collection = st.sidebar.selectbox(
            "Pilih koleksi:",
            options=["-- Pilih Koleksi --"] + collection_names,
            index=0, # Default to "Pilih Koleksi"
            key="collection_selector"
        )

        if selected_collection != "-- Pilih Koleksi --":
            st.session_state.current_collection = selected_collection
            st.sidebar.success(f"Koleksi aktif diatur ke: **{selected_collection}**")
        else:
            if 'current_collection' in st.session_state:
                del st.session_state.current_collection # Hapus koleksi aktif jika tidak ada yang dipilih
            st.sidebar.info("Tidak ada koleksi yang dipilih.")
            
    else:
        st.sidebar.info("Belum ada koleksi di ChromaDB.")
        if 'current_collection' in st.session_state:
            del st.session_state.current_collection # Hapus koleksi aktif jika tidak ada yang dipilih

except Exception as e:
    st.sidebar.error(f"Gagal memuat koleksi ChromaDB: {e}")
    if 'current_collection' in st.session_state:
            del st.session_state.current_collection

st.divider()

# Bagian Chatting
st.header("2. Ajukan Pertanyaan")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Ensure current_collection is always initialized even if it's None
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = None 

if st.session_state.current_collection:
    st.info(f"Anda sedang chatting dengan dokumen di koleksi: **{st.session_state.current_collection}**")
else:
    st.warning("Harap unggah dan proses PDF atau pilih koleksi yang sudah ada untuk memulai chatting.")

user_query = st.text_input("Pertanyaan Anda:", key="user_query_input")

if user_query and st.button("Kirim Pertanyaan"):
    if st.session_state.current_collection:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("Mencari jawaban..."):
            try:
                # 1. Retrieve
                retrieved_docs = retrieve_documents(user_query, st.session_state.current_collection)
                
                if retrieved_docs:
                    # 2. Augment (Create RAG prompt)
                    rag_prompt = create_rag_prompt(user_query, retrieved_docs)
                    
                    # 3. Generate
                    ai_response = generate_response(rag_prompt)
                    
                    if ai_response:
                        st.session_state.chat_history.append({"role": "ai", "content": ai_response})
                    else:
                        st.session_state.chat_history.append({"role": "ai", "content": "Maaf, saya tidak dapat menghasilkan respons."})
                else:
                    st.session_state.chat_history.append({"role": "ai", "content": "Maaf, saya tidak menemukan informasi relevan dalam dokumen yang diunggah untuk pertanyaan ini."})
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan RAG: {e}")
                st.session_state.chat_history.append({"role": "ai", "content": f"Maaf, terjadi kesalahan: {e}"})
    else:
        st.warning("Harap unggah dan proses PDF atau pilih koleksi yang sudah ada.")

# Tampilkan riwayat chat
st.header("Riwayat Chat")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**Anda:** {chat['content']}")
    else:
        st.markdown(f"**AI:** {chat['content']}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Tentang Aplikasi")
st.sidebar.markdown(
    """
    Aplikasi ini memungkinkan Anda mengunggah file PDF, memprosesnya menjadi
    chunks teks, menyimpannya di ChromaDB, dan kemudian melakukan
    Retrieval Augmented Generation (RAG) untuk berinteraksi dengan dokumen
    menggunakan model bahasa dari OpenAI.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Folder ChromaDB")
st.sidebar.markdown(
    """
    Data ChromaDB akan disimpan di folder `./chroma_db` di direktori yang sama
    dengan skrip ini.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Embedding")
st.sidebar.markdown(
    """
    Menggunakan `all-MiniLM-L6-v2` dari `sentence-transformers` untuk membuat embedding.
    """
)
