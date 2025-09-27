"""
RAG Resume Search Chatbot
--------------------------------
Single-file Streamlit app that lets a user upload bulk PDF resumes, builds a FAISS vectorstore
(using OpenAI embeddings), and provides a retrieval-augmented QA interface to search for
particular candidates and ask questions about resumes.

Requirements (pip):
  pip install streamlit langchain faiss-cpu langchain-google-genai PyPDF2

How to run:
  1. Set environment variable GEMINI_API_KEY (or create a .env file with GEMINI_API_KEY=...)
  2. streamlit run rag_resume_bot.py

Notes:
 - This is a minimal, opinionated implementation using Google Gemini for embeddings and LLM.
 - Uploaded PDFs are temporarily saved to a local `uploads/` folder and the vectorstore is persisted
   in `faiss_store/` so you can reuse indices between runs.

"""

import os
import tempfile
import shutil
from typing import List, Optional

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Config
UPLOAD_DIR = "uploads"
STORE_DIR = "faiss_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"  # Google Generative AI embedding model
LLM_MODEL = "gemini-1.0-pro"  # Google Generative AI model

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STORE_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Resume Bot", layout="wide")
st.title("📄 RAG Resume Bot — bulk PDF resumes → semantic search & QA")

st.markdown(
    "Upload one or many PDF resumes. The app will extract text, split into chunks, embed, and store in a local FAISS index."
)

# ---- Helpers ----

def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded Streamlit files to UPLOAD_DIR and return file paths."""
    saved_paths = []
    for uploaded in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def pdf_to_documents(file_paths: List[str]) -> List[Document]:
    """Load PDFs to LangChain Document objects, adding metadata with source filename and page."""
    docs: List[Document] = []
    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            # Each page is already a Document with page content. Add filename to metadata.
            for i, p in enumerate(pages, start=1):
                p.metadata["source_file"] = os.path.basename(path)
                p.metadata["page"] = i
                docs.append(p)
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into chunks for embedding. Keeps metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    new_docs: List[Document] = []
    for d in docs:
        splits = splitter.split_text(d.page_content)
        for i, chunk in enumerate(splits):
            metadata = dict(d.metadata)
            metadata["chunk_id"] = f"{metadata.get('source_file','unknown')}_p{metadata.get('page','0')}_c{i}"
            new_docs.append(Document(page_content=chunk, metadata=metadata))
    return new_docs


def get_embedding_client():
    """Create a GoogleGenerativeAIEmbeddings object. Make sure GEMINI_API_KEY is set."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def create_or_load_vectorstore(docs: List[Document], persist_directory: str = STORE_DIR) -> FAISS:
    """Create or update a FAISS index with passed documents. If an index exists, it will be loaded and extended."""
    embeddings = get_embedding_client()
    if os.path.exists(os.path.join(persist_directory, "index.faiss")):
        # load existing store and add docs
        store = FAISS.load_local(persist_directory, embeddings)
        if docs:
            store.add_documents(docs)
            store.save_local(persist_directory)
        return store
    else:
        store = FAISS.from_documents(docs, embeddings)
        store.save_local(persist_directory)
        return store


def build_qa_chain(vectorstore: FAISS):
    """Build a RetrievalQA chain using Google Generative AI LLM and the provided vectorstore retriever."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa


def semantic_search_candidates(vectorstore: FAISS, name_query: str, k: int = 5) -> List[Document]:
    """Run a semantic search for candidate name queries and return matching Document chunks (with metadata)."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.get_relevant_documents(name_query)


# ---- Streamlit UI ----

st.sidebar.header("Index Controls")
uploaded_files = st.sidebar.file_uploader("Upload PDF resumes (multiple)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    if st.sidebar.button("Ingest uploads and update index"):
        with st.spinner("Saving uploads and extracting text..."):
            saved_paths = save_uploaded_files(uploaded_files)
            raw_docs = pdf_to_documents(saved_paths)
            st.write(f"Loaded {len(raw_docs)} pages from {len(saved_paths)} PDF(s)")
            chunks = chunk_documents(raw_docs)
            st.write(f"Split into {len(chunks)} chunks")
            with st.spinner("Indexing embeddings (this may take a while)..."):
                store = create_or_load_vectorstore(chunks)
                st.success("Index updated and saved to disk.")

if st.sidebar.button("Clear index & uploads (danger)"):
    if st.sidebar.checkbox("Confirm delete all stored data"):
        try:
            shutil.rmtree(STORE_DIR)
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            os.makedirs(STORE_DIR, exist_ok=True)
            st.sidebar.success("Cleared index and uploads")
        except Exception as e:
            st.sidebar.error(f"Error clearing data: {e}")

# Main operations
st.header("Find a candidate / Ask about resumes")

# Load index if present
embeddings_present = os.path.exists(os.path.join(STORE_DIR, "index.faiss"))
if not embeddings_present:
    st.info("No index found yet. Upload PDFs in the left sidebar and click 'Ingest uploads and update index'.")

col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("Search query / candidate name / question", placeholder="e.g. 'Find candidate Rahim Khan' or 'experience of Priya in Azure'")
    ask = st.button("Ask")

with col2:
    name_search_only = st.checkbox("Name-search mode (prioritize name matches)")
    top_k = st.number_input("Top K results", min_value=1, max_value=20, value=5)

if ask and query:
    if not embeddings_present:
        st.error("No index available. Please ingest resumes first.")
    else:
        store = FAISS.load_local(STORE_DIR, get_embedding_client())

        if name_search_only:
            # Use semantic search but emphasize exact name matches by doing a quick text filter first
            sem_results = semantic_search_candidates(store, query, k=top_k)
            # Additionally add exact substring matches from metadata (strong signal for candidate name)
            exact_hits = []
            for doc in sem_results:
                if query.lower() in doc.page_content.lower() or query.lower() in (doc.metadata.get("source_file","") .lower()):
                    exact_hits.append(doc)
            results = exact_hits + [d for d in sem_results if d not in exact_hits]
            results = results[:top_k]

            if not results:
                st.warning("No candidate-like results found. Try a broader query or uncheck name-search mode.")
            else:
                st.success(f"Found {len(results)} relevant chunks")
                for r in results:
                    st.markdown("---")
                    st.write(f"**Source file:** {r.metadata.get('source_file')} | **Page:** {r.metadata.get('page')} | **Chunk:** {r.metadata.get('chunk_id')}")
                    snippet = r.page_content
                    st.write(snippet[:1000] + ("..." if len(snippet) > 1000 else ""))
                    if st.button(f"Open full resume: {r.metadata.get('source_file')}", key=r.metadata.get('chunk_id')):
                        # Show the entire PDF using streamlit's components
                        pdf_path = os.path.join(UPLOAD_DIR, r.metadata.get('source_file'))
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        st.download_button("Download full PDF", data=pdf_bytes, file_name=r.metadata.get('source_file'))

        else:
            # Retrieval QA for general queries
            qa = build_qa_chain(store)
            with st.spinner("Running retrieval + LLM..."):
                try:
                    resp = qa(query)
                    answer = resp.get("result") if isinstance(resp, dict) else str(resp)
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"LLM failed: {e}. Showing retrieved context instead.")
                    # Fallback to show retrieved documents
                    docs = store.similarity_search(query, k=top_k)
                    st.markdown("### Retrieved Context")
                    for d in docs:
                        st.write(f"**{d.metadata.get('source_file')} (page {d.metadata.get('page')})**")
                        st.write(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

            st.markdown("### Source chunks used")
            src_docs = resp.get("source_documents", []) if isinstance(resp, dict) and 'resp' in locals() else []
            for d in src_docs:
                st.write(f"- {d.metadata.get('source_file')} (page {d.metadata.get('page')}) — {d.metadata.get('chunk_id')}")

# Bonus: quick candidate discovery UI
st.sidebar.header("Quick candidate lookup")
candidate_name = st.sidebar.text_input("Candidate name to lookup")
if st.sidebar.button("Find candidate"):
    if not os.path.exists(os.path.join(STORE_DIR, "index.faiss")):
        st.sidebar.error("No index found. Ingest PDFs first.")
    else:
        store = FAISS.load_local(STORE_DIR, get_embedding_client())
        hits = semantic_search_candidates(store, candidate_name, k=int(top_k))
        st.sidebar.write(f"Found {len(hits)} chunks")
        for h in hits:
            st.sidebar.write(f"{h.metadata.get('source_file')} — p{h.metadata.get('page')} — {h.metadata.get('chunk_id')}")

st.markdown("---")
st.caption("This demo stores the FAISS index in a local folder and uses Google Generative AI embeddings + LLM. For production use, consider secure storage, larger embedding/LLM models, and privacy/consent for resume data.")

# EOF
