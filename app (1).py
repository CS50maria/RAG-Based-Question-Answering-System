import os
import numpy as np
import streamlit as st
import faiss, PyPDF2, docx
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚")
st.title("📚 Document Q&A Assistant")
st.caption("CS4106 — Foundations of Generative AI")

# ---- API key comes from the secret GROQ_API_KEY (never typed in the app) ---
api_key = os.environ.get("GROQ_API_KEY")

# ---- Settings ---
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["llama-3.3-70b-versatile",
                                   "openai/gpt-oss-120b"])
    top_k = st.slider("Chunks to retrieve (k)", 1, 5, 3)
    if st.button("Clear knowledge base"):
        for key in ("chunks", "sources", "index"):
            st.session_state.pop(key, None)
        st.success("Knowledge base cleared.")


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

def read_file(f):
    if f.name.endswith(".pdf"):
        return "".join(p.extract_text() or "" for p in PyPDF2.PdfReader(f).pages)
    if f.name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(f).paragraphs)
    return f.read().decode("utf-8", "ignore")

def chunk_text(text, size=150, overlap=30):
    words = text.split(); out = []
    for i in range(0, len(words), size - overlap):
        out.append(" ".join(words[i:i + size]))
        if i + size >= len(words): break
    return out

def build_index(chunks):
    emb = embedder.encode(chunks)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(emb.shape[1]); idx.add(emb.astype("float32"))
    return idx

def search(idx, chunks, sources, query, k):
    q = embedder.encode([query]); q = q / np.linalg.norm(q, axis=1, keepdims=True)
    scores, ids = idx.search(q.astype("float32"), k)
    return [(chunks[i], sources[i], float(s))
            for s, i in zip(scores[0], ids[0]) if 0 <= i < len(chunks)]

# ---- Upload & build the knowledge base ---
files = st.file_uploader("Upload PDF / DOCX / TXT",
                         type=["pdf", "docx", "txt"], accept_multiple_files=True)
if files and st.button("Build knowledge base"):
    chunks, sources = [], []
    for f in files:
        for c in chunk_text(read_file(f)):
            chunks.append(c)
            sources.append(f.name)      # remember which file each chunk came from
    if not chunks:
        st.error("No text could be extracted (a scanned PDF has no text). Try a text-based file.")
    else:
        st.session_state.chunks = chunks
        st.session_state.sources = sources
        st.session_state.index = build_index(chunks)
        st.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s).")

# ---- Ask a question ---
question = st.text_input("Ask a question about your documents:")
if st.button("Answer") and question:
    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it as a secret and restart the app.")
    elif "index" not in st.session_state:
        st.error("Upload a document and click 'Build knowledge base' first.")
    else:
        hits = search(st.session_state.index, st.session_state.chunks,
                      st.session_state.sources, question, top_k)
        context = "\n\n".join(d for d, _, _ in hits)
        prompt = ("Answer using ONLY the context. If it is not there, say you "
                  f"don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:")
        client = Groq(api_key=api_key)
        r = client.chat.completions.create(model=model,
                messages=[{"role": "user", "content": prompt}])
        st.subheader("Answer")
        st.write(r.choices[0].message.content)
        with st.expander("Sources (retrieved chunks)"):
            for i, (d, src, s) in enumerate(hits, 1):
                st.markdown(f"**Chunk {i}** · `{src}` · score {s:.2f}")
                st.write(d)
