import streamlit as st
from langchain_community.document_loaders import PyPDFLoader # Import yang benar
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone # Import yang benar
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pinecone
import os

# --- Konfigurasi Awal Streamlit ---
st.set_page_config(page_title="Chatbot Tanya Jawab Dokumen", layout="wide")
st.title("Chatbot Tanya Jawab Dokumen dengan Pinecone dan OpenAI")

# --- Sidebar untuk API Keys dan Konfigurasi ---
with st.sidebar:
    st.header("Konfigurasi API dan Pinecone")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    pinecone_environment = st.text_input("Pinecone Environment", value="us-west1-gcp") # Ganti dengan environment Pinecone Anda
    pinecone_index_name = st.text_input("Nama Index Pinecone", value="chatbot-index") # Ganti dengan nama index Anda

    st.header("Upload Dokumen")
    uploaded_file = st.file_uploader("Upload file PDF Anda", type="pdf")

    if st.button("Proses Dokumen dan Inisialisasi Chatbot"):
        if openai_api_key and pinecone_api_key and uploaded_file:
            
            pinecone.init(
                api_key=pinecone_api_key,
                environment=pinecone_environment
            )

            with st.spinner("Memproses dokumen dan membuat embedding..."):
                with open("temp_doc.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader("temp_doc.pdf")
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150,
                    length_function=len,
                )
                texts = text_splitter.split_documents(documents)

                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key) 
                
                if pinecone_index_name not in pinecone.list_indexes():
                    pinecone.create_index(
                        name=pinecone_index_name,
                        dimension=1536,
                        metric='cosine'
                    )
                
                docsearch = Pinecone.from_documents(
                    texts, 
                    embeddings, 
                    index_name=pinecone_index_name
                )
                
                st.session_state.vectorstore = docsearch
                st.session_state.embeddings = embeddings
                st.success("Dokumen berhasil diproses dan disimpan ke Pinecone!")
                os.remove("temp_doc.pdf")
        else:
            st.error("Harap masukkan semua API Key dan upload dokumen.")

# --- Inisialisasi State Awal ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# --- Inisialisasi Conversational Chain setelah vectorstore siap ---
if st.session_state.vectorstore and st.session_state.conversation is None:
    if openai_api_key:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=openai_api_key) 
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=memory
        )
        st.success("Chatbot siap digunakan!")
    else:
        st.error("Harap masukkan OpenAI API Key untuk menginisialisasi chatbot.")

# --- Tampilan Riwayat Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Pengguna ---
if prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.conversation:
        with st.spinner("Mencari jawaban..."):
            response = st.session_state.conversation({"question": prompt})
            answer = response["answer"]
            st.session_state.chat_history.append((prompt, answer))
            
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            st.markdown("Harap inisialisasi chatbot terlebih dahulu dengan mengupload dokumen dan memasukkan API key.")
