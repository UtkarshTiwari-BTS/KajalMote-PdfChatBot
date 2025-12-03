import streamlit as st
import os
import logging

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
    # FIX: normalize vectordb.embedding_function safely
    # -------------------------------------------------
    try:
        vdb = st.session_state.vectordb
        if vdb is not None:
            orig = getattr(vdb, "embedding_function", None)

            def _wrapped_embedding(x):
                """
                Safe embedding handler. Works for:
                - callable embedding_function
                - DummyEmbeddings object (embed_query, embed_documents)
                - vectordb embed_query / embed_documents
                """
                # 1) If orig is callable
                if callable(orig):
                    try:
                        return orig(x)
                    except TypeError:
                        if isinstance(x, str):
                            return orig([x])[0]
                        return orig(list(x))

                # 2) orig is an embedding object
                if orig is not None:
                    # embed_query
                    if hasattr(orig, "embed_query") and callable(orig.embed_query):
                        try:
                            if isinstance(x, str):
                                return orig.embed_query(x)
                            return [orig.embed_query(t) for t in x]
                        except Exception:
                            pass

                    # embed_documents
                    if hasattr(orig, "embed_documents") and callable(orig.embed_documents):
                        try:
                            if isinstance(x, str):
                                return orig.embed_documents([x])[0]
                            return orig.embed_documents(list(x))
                        except Exception:
                            pass

                # 3) Try vectordb-level embed_query
                if hasattr(vdb, "embed_query") and callable(vdb.embed_query):
                    try:
                        if isinstance(x, str):
                            return vdb.embed_query(x)
                        return [vdb.embed_query(t) for t in x]
                    except Exception:
                        pass

                # 4) vectordb embed_documents
                if hasattr(vdb, "embed_documents") and callable(vdb.embed_documents):
                    try:
                        if isinstance(x, str):
                            return vdb.embed_documents([x])[0]
                        return vdb.embed_documents(list(x))
                    except Exception:
                        pass

                raise TypeError("Unable to produce embeddings for query input.")

            # assign wrapped embedding function
            vdb.embedding_function = _wrapped_embedding

    except Exception as e:
        logging.exception("Failed normalizing embedding function: %s", e)


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
