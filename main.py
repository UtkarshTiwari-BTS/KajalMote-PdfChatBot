import warnings
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

import os
from pdfutils import extract_pdf_text, chunk_text
from embeddings import build_vector_store
from rag import search, ask_llm

if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    print("\n Extracting PDF text...")
    text = extract_pdf_text(pdf_path)   #It will return single string containing whole PDF text

    print(" Chunking text...")
    chunks = chunk_text(text)           # list of text chunks
    print("Chunk Length :",len(chunks))

    print(" Building FAISS vector DB using TF-IDF embeddings...")
    vectordb = build_vector_store(chunks)

    print("\n READY! Ask questions about the PDF.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        relevant_docs = search(vectordb, query, k=5)
        answer = ask_llm(relevant_docs, query)

        print("Bot:", answer)
