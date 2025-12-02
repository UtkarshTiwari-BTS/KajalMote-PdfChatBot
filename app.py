import streamlit as st
import os

from pdfutils import extract_pdf_text, chunk_text
from embeddings import build_vector_store
from rag import search, ask_llm

# -------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------
st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")

# Initialize session state
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------------------------
# Sidebar: Upload PDF
# -------------------------------------------------
st.sidebar.header("Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_pdf:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    st.sidebar.success("PDF uploaded successfully!")

    # Extract PDF text
    text = extract_pdf_text("uploaded.pdf")
    chunks = chunk_text(text)

    st.session_state.vectordb = build_vector_store(chunks)
    st.sidebar.success("Vector DB Built! Chat Ready ✔")


# -------------------------------------------------
# Main Chat Interface
# -------------------------------------------------
st.title(" PDF Question Answering Chatbot")

if st.session_state.vectordb is None:
    st.info(" Upload a PDF to begin.")
else:

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # Chat input box
    user_query = st.chat_input("Ask a question about the PDF...")

    if user_query:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Retrieve relevant sections
        results = search(st.session_state.vectordb, user_query, k=3)

        # LLM answer
        answer = ask_llm(results, user_query)

        # Add bot answer to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Display bot answer immediately
        with st.chat_message("assistant"):
            st.write(answer)

