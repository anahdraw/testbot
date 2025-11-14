import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import os

st.set_page_config(layout="wide")
st.title("ChromaDB Local to TryChroma (Cloud) Uploader")
st.markdown("---")

# --- Sidebar Configuration ---
st.sidebar.header("TryChroma Cloud Settings")
trychroma_api_key = st.sidebar.text_input(
    "TryChroma API Key/Token", 
    type="password", 
    help="Dapatkan dari dashboard TryChroma Anda."
)
trychroma_client_url = st.sidebar.text_input(
    "TryChroma Client URL",
    value="https://api.trychroma.com", # Default URL, sesuaikan jika berbeda
    help="URL endpoint ChromaDB Cloud Anda."
)

st.sidebar.markdown("---")
st.sidebar.header("Local ChromaDB Settings")
local_chroma_path = st.sidebar.text_input(
    "Local ChromaDB Path", 
    value="./chroma_db", 
    help="Path ke folder ChromaDB lokal Anda. (Misal: ./chroma_db)"
)

# --- Initialize Embedding Model (used for consistency) ---
@st.cache_resource
def load_embedding_model():
    # Use the same embedding model as your local setup for consistency
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

# Create a ChromaDB embedding function for cloud
class SentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: embedding_functions.Documents) -> embedding_functions.Embeddings:
        return self.model.encode(input).tolist()

chroma_embedding_function = SentenceTransformerEmbeddingFunction('all-MiniLM-L6-v2')


# --- Main Application Logic ---

if not trychroma_api_key or not trychroma_client_url:
    st.warning("Harap masukkan API Key dan Client URL TryChroma di sidebar untuk melanjutkan.")
    st.stop()

st.header("1. Hubungkan ke ChromaDB Lokal")
local_chroma_client = None
try:
    if os.path.exists(local_chroma_path):
        local_chroma_client = chromadb.PersistentClient(path=local_chroma_path)
        st.success(f"Berhasil terhubung ke ChromaDB lokal di: `{local_chroma_path}`")
        local_collections = local_chroma_client.list_collections()
        if local_collections:
            st.write("Koleksi lokal yang ditemukan:")
            local_collection_names = [col.name for col in local_collections]
            selected_local_collection_name = st.selectbox(
                "Pilih koleksi lokal untuk diunggah:", 
                options=["-- Pilih Koleksi --"] + local_collection_names
            )
        else:
            st.info("Tidak ada koleksi di ChromaDB lokal Anda.")
            selected_local_collection_name = None
    else:
        st.error(f"Folder ChromaDB lokal tidak ditemukan di: `{local_chroma_path}`")
        st.info("Pastikan Anda sudah menjalankan aplikasi RAG sebelumnya untuk membuat database lokal.")
        selected_local_collection_name = None
except Exception as e:
    st.error(f"Gagal terhubung ke ChromaDB lokal: {e}")
    st.info("Pastikan path sudah benar dan folder tidak terkunci.")
    selected_local_collection_name = None


st.header("2. Hubungkan ke TryChroma Cloud")
cloud_chroma_client = None
try:
    cloud_chroma_client = chromadb.HttpClient(
        host=trychroma_client_url.replace("https://", "").replace("http://", ""), # Hapus protokol dari host
        port=443 if "https" in trychroma_client_url else 80, # Sesuaikan port jika diperlukan
        ssl=True if "https" in trychroma_client_url else False,
        headers={"X-Chroma-Token": trychroma_api_key}
    )
    cloud_chroma_client.heartbeat() # Test connection
    st.success(f"Berhasil terhubung ke TryChroma Cloud di: `{trychroma_client_url}`")
    cloud_collections = cloud_chroma_client.list_collections()
    if cloud_collections:
        st.write("Koleksi yang sudah ada di TryChroma Cloud:")
        for col in cloud_collections:
            st.code(col.name)
except Exception as e:
    st.error(f"Gagal terhubung ke TryChroma Cloud: {e}")
    st.info("Pastikan API Key, Client URL sudah benar, dan koneksi internet Anda stabil.")
    st.stop() # Stop further execution if cloud connection fails

st.header("3. Unggah Koleksi ke TryChroma Cloud")
if selected_local_collection_name and selected_local_collection_name != "-- Pilih Koleksi --":
    new_cloud_collection_name = st.text_input(
        f"Nama koleksi di Cloud untuk `{selected_local_collection_name}`:",
        value=f"{selected_local_collection_name}_cloud"
    )
    overwrite_existing = st.checkbox("Timpa koleksi di Cloud jika sudah ada (PERINGATAN: Ini akan menghapus data yang ada!)")

    if st.button(f"Mulai Unggah Koleksi '{selected_local_collection_name}'"):
        if not new_cloud_collection_name:
            st.error("Nama koleksi di Cloud tidak boleh kosong.")
        else:
            with st.spinner(f"Mengambil data dari koleksi lokal '{selected_local_collection_name}'..."):
                try:
                    local_collection = local_chroma_client.get_collection(name=selected_local_collection_name)
                    # Fetch all data from the local collection
                    all_data = local_collection.get(
                        ids=local_collection.get()['ids'], # Get all IDs
                        include=['documents', 'embeddings', 'metadatas']
                    )

                    documents = all_data['documents']
                    embeddings = all_data['embeddings']
                    metadatas = all_data['metadatas']
                    ids = all_data['ids']

                    st.success(f"Berhasil mengambil {len(documents)} item dari ChromaDB lokal.")

                except Exception as e:
                    st.error(f"Gagal mengambil data dari koleksi lokal: {e}")
                    st.stop()

            with st.spinner(f"Mengunggah data ke TryChroma Cloud di koleksi '{new_cloud_collection_name}'..."):
                try:
                    # Check if collection exists in cloud and handle overwrite
                    cloud_collection_names = [col.name for col in cloud_chroma_client.list_collections()]
                    if new_cloud_collection_name in cloud_collection_names:
                        if overwrite_existing:
                            st.warning(f"Menghapus koleksi '{new_cloud_collection_name}' yang sudah ada di Cloud...")
                            cloud_chroma_client.delete_collection(name=new_cloud_collection_name)
                            st.success("Koleksi lama dihapus.")
                        else:
                            st.error(f"Koleksi '{new_cloud_collection_name}' sudah ada di Cloud. Centang 'Timpa koleksi...' atau gunakan nama baru.")
                            st.stop()
                    
                    # Create the new collection in cloud
                    cloud_collection = cloud_chroma_client.create_collection(
                        name=new_cloud_collection_name,
                        embedding_function=chroma_embedding_function # Penting: Gunakan embedding function yang sama
                    )

                    # Add data in chunks to avoid API limits for large collections
                    chunk_size = 500 # Adjust based on your API limits and network
                    for i in range(0, len(documents), chunk_size):
                        st.info(f"Mengunggah chunk {i//chunk_size + 1}/{(len(documents)//chunk_size)+1}...")
                        chunk_docs = documents[i:i + chunk_size]
                        chunk_embeds = embeddings[i:i + chunk_size]
                        chunk_metadatas = metadatas[i:i + chunk_size]
                        chunk_ids = ids[i:i + chunk_size]

                        cloud_collection.add(
                            documents=chunk_docs,
                            embeddings=chunk_embeds,
                            metadatas=chunk_metadatas,
                            ids=chunk_ids
                        )
                    
                    st.success(f"Berhasil mengunggah koleksi '{new_cloud_collection_name}' ke TryChroma Cloud!")
                    st.balloons()

                except Exception as e:
                    st.error(f"Gagal mengunggah ke TryChroma Cloud: {e}")
                    st.info("Periksa API Key, URL, dan pastikan tidak ada batasan ukuran data atau rate limit dari TryChroma.")
else:
    st.info("Harap pilih koleksi lokal di langkah 1 untuk memulai proses unggah.")

st.markdown("---")
st.sidebar.markdown("### Cara Kerja")
st.sidebar.markdown(
    """
    Aplikasi ini membaca data (dokumen, embedding, metadata, dan ID) dari 
    koleksi ChromaDB lokal Anda. Kemudian, ia menghubungkan ke layanan 
    TryChroma Cloud Anda menggunakan API Key dan URL yang Anda berikan. 
    Terakhir, ia membuat koleksi baru di Cloud dan mengunggah semua data 
    dari koleksi lokal ke sana.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Penting:")
st.sidebar.markdown(
    """
    *   Pastikan Anda menggunakan **model embedding yang sama** baik untuk 
        ChromaDB lokal maupun saat mengunggah ke Cloud. Di sini digunakan 
        `all-MiniLM-L6-v2`.
    *   Mengunggah koleksi besar mungkin memakan waktu dan tergantung pada 
        batasan API dari TryChroma. Data diunggah dalam chunk.
    *   Menimpa koleksi akan **menghapus semua data** yang sudah ada di 
        koleksi Cloud dengan nama yang sama.
    """
)
