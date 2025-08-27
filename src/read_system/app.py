import streamlit as st
from embedder import LocalEmbedder
from vector_store import VectorStore
from llm_interface import Client
from chunker import chunk_text  
import tempfile
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        doc = fitz.open(tmp.name)
        text = "\n".join([page.get_text() for page in doc])
        return text

def extract_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text()
    except Exception:
        return "[Error retrieving or parsing web page.]"


st.set_page_config(page_title="RAG QA Reader", layout="wide")
st.title("📄 Document Q&A with GPT-4o")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "recent_chunk_ids" not in st.session_state:
    st.session_state.recent_chunk_ids = set()

gpt = Client()
embedder = LocalEmbedder()
store = VectorStore(dim=384)

uploaded_text = st.text_area("Paste your document content here:", height=300)
question = st.text_input("Ask a question about the content:")

col1, col2 = st.columns([1, 1])
with col1:
    run = st.button("🔍 Run RAG")
with col2:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.recent_chunk_ids = set()

if run and uploaded_text and question:
    with st.spinner("Processing..."):
        chunks = chunk_text(uploaded_text)
        vectors = embedder.embed_many(chunks)
        metadatas = [{"content": ch, "chunk_id": i} for i, ch in enumerate(chunks)]
        store.add(vectors, metadatas)

        query_vec = embedder.embed(question)
        top_chunks = store.search(query_vec, top_k=5)

        # 合併歷史用過的 chunk_id，避免重複
        selected_chunk_ids = {meta["chunk_id"] for _, meta in top_chunks}
        st.session_state.recent_chunk_ids.update(selected_chunk_ids)
        all_chunk_ids = sorted(st.session_state.recent_chunk_ids)
        selected_chunks = [meta for meta in store.metadata if meta["chunk_id"] in all_chunk_ids]

        context_text = "\n".join([f"[C{i+1}] {item['content']}" for i, item in enumerate(selected_chunks)])

        system_prompt = (
            "Answer using only the passages in Provided Context; if the context is insufficient say so,"
            " merge overlapping details into a single coherent explanation without repeating sentences,"
            " and include a short quote or the passage tag (e.g., [C2]) for precise facts while avoiding outside information."
        )

        user_query = f"Provided Context:\n{context_text}\n\nQuestion: {question}"

        answer, updated_history = gpt.chat(system_prompt, user_query, st.session_state.chat_history)
        st.session_state.chat_history = updated_history

        st.markdown("### 💬 Answer")
        st.write(answer)

        st.markdown("---")
        st.markdown("### 🔍 Top Retrieved Chunks")
        for score, meta in top_chunks:
            st.markdown(f"**Score: {score:.4f}**\n\n{meta['content']}")

        st.markdown("---")
        st.markdown("### 🧠 Chat History")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Assistant:** {msg['content']}")