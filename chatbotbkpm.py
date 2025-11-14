import streamlit as st
import openai
import requests
import os
import io
import PyPDF2

# --- Fungsi Bantuan ---

def extract_text_from_pdf(file_bytes):
    """Mengekstrak teks dari byte file PDF yang diunggah."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    except Exception as e:
        st.error(f"Gagal mengekstrak teks dari PDF: {e}")
        return None

def review_contract(contract_text, api_key):
    """Mengirim teks kontrak ke OpenAI API untuk ditinjau."""
    if not contract_text or not contract_text.strip():
        st.warning("Tidak ada teks yang dapat dianalisis dari file yang diunggah.")
        return None
    try:
        openai.api_key = api_key # Mengatur kunci API sebelum melakukan panggilan
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Anda adalah asisten hukum ahli yang berspesialisasi dalam peninjauan kontrak. Tinjau dokumen berikut secara komprehensif, identifikasi potensi risiko, klausul yang tidak jelas, dan area yang mungkin memerlukan negosiasi lebih lanjut. Berikan ringkasan, poin-poin penting, dan saran dalam format yang jelas dan mudah dibaca."},
                {"role": "user", "content": contract_text}
            ]
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        st.error("Kunci API OpenAI tidak valid atau salah. Harap periksa kembali di sidebar.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat berkomunikasi dengan OpenAI: {e}")
        return None

def send_to_n8n(decision, contract_name, review_summary, webhook_url):
    """Mengirim keputusan dan data relevan ke webhook n8n."""
    try:
        payload = {
            "decision": decision,
            "contract_name": contract_name,
            "review_summary": review_summary
        }
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Gagal mengirim data ke n8n: Pastikan URL Webhook benar dan dapat diakses. Error: {e}")
        return False

# --- Inisialisasi Status Sesi (Session State) ---
# Diperlukan untuk menyimpan nilai antar interaksi pengguna
if 'config_set' not in st.session_state:
    st.session_state.config_set = False
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'n8n_webhook_url' not in st.session_state:
    st.session_state.n8n_webhook_url = ""
if 'review_result' not in st.session_state:
    st.session_state.review_result = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'processing' not in st.session_state:
    st.session_state.processing = False


# --- Antarmuka Pengguna Streamlit ---
st.set_page_config(page_title="Tinjauan Kontrak AI", layout="wide")

# ==================== SIDEBAR UNTUK KONFIGURASI ====================
st.sidebar.header("⚙️ Konfigurasi")
st.sidebar.warning("Kunci API dan URL Webhook Anda tidak disimpan secara permanen. Anda perlu memasukkannya setiap kali memuat ulang halaman.")

api_key_input = st.sidebar.text_input(
    "1. Masukkan Kunci API OpenAI Anda",
    type="password",
    placeholder="sk-...",
    help="Dapatkan kunci API Anda dari platform.openai.com",
    value=st.session_state.openai_api_key
)

webhook_url_input = st.sidebar.text_input(
    "2. Masukkan URL Webhook n8n Anda",
    placeholder="https://n8n.anda.com/webhook/...",
    help="Salin URL dari node Webhook di alur kerja n8n Anda.",
    value=st.session_state.n8n_webhook_url
)

if st.sidebar.button("Simpan Konfigurasi"):
    if api_key_input.startswith("sk") and ("http") in webhook_url_input:
        st.session_state.openai_api_key = api_key_input
        st.session_state.n8n_webhook_url = webhook_url_input
        st.session_state.config_set = True
        st.sidebar.success("Konfigurasi berhasil disimpan!")
    else:
        st.sidebar.error("Harap masukkan Kunci API dan URL Webhook yang valid.")
        st.session_state.config_set = False

# ==================== APLIKASI UTAMA ====================
st.title("📄 Tinjauan Dokumen Kontrak Komprehensif dengan AI")

# Tampilkan aplikasi utama HANYA JIKA konfigurasi sudah diatur
if not st.session_state.config_set:
    st.info("Harap masukkan Kunci API OpenAI dan URL Webhook di sidebar untuk memulai.")
    st.stop() # Menghentikan eksekusi jika belum dikonfigurasi

st.markdown("Unggah dokumen kontrak Anda (.pdf, .txt, .md) untuk mendapatkan analisis mendalam dari OpenAI.")

uploaded_file = st.file_uploader("Pilih file dokumen (.pdf, .txt, .md)", type=['pdf', 'txt', 'md'])

if uploaded_file is not None:
    if st.button("Tinjau Dokumen", disabled=st.session_state.processing):
        with st.spinner("Membaca dan menganalisis kontrak... Harap tunggu sebentar."):
            st.session_state.processing = True
            contract_text = None
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                contract_bytes = uploaded_file.read()

                if file_extension == ".pdf":
                    contract_text = extract_text_from_pdf(contract_bytes)
                else:
                    contract_text = contract_bytes.decode("utf-8")

                if contract_text:
                    st.session_state.file_name = uploaded_file.name
                    # Kirim kunci API yang disimpan di session state ke fungsi
                    st.session_state.review_result = review_contract(
                        contract_text,
                        st.session_state.openai_api_key
                    )
                else:
                    st.session_state.review_result = None

            except Exception as e:
                st.error(f"Gagal memproses file: {e}")
                st.session_state.review_result = None
            finally:
                st.session_state.processing = False

if st.session_state.review_result:
    st.subheader("Hasil Tinjauan AI")
    st.markdown(st.session_state.review_result)

    st.subheader("Tindakan")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Setujui Dokumen", type="primary"):
            with st.spinner("Mengirim persetujuan..."):
                success = send_to_n8n(
                    "disetujui",
                    st.session_state.file_name,
                    st.session_state.review_result,
                    st.session_state.n8n_webhook_url # Kirim URL webhook
                )
                if success:
                    st.success(f"Dokumen '{st.session_state.file_name}' telah disetujui dan notifikasi telah dikirim.")
                    st.session_state.review_result = None
                    st.session_state.file_name = None

    with col2:
        if st.button("❌ Tolak Dokumen"):
            with st.spinner("Mengirim penolakan..."):
                success = send_to_n8n(
                    "ditolak",
                    st.session_state.file_name,
                    st.session_state.review_result,
                    st.session_state.n8n_webhook_url # Kirim URL webhook
                )
                if success:
                    st.success(f"Dokumen '{st.session_state.file_name}' telah ditolak dan notifikasi telah dikirim.")
                    st.session_state.review_result = None
                    st.session_state.file_name = None
