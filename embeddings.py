from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.embeddings.base import Embeddings

def build_vector_store(chunks):
    docs = [Document(page_content=ch) for ch in chunks]
    texts = [d.page_content for d in docs]

    #print("→ Vectorizing using TF-IDF...")
    print(" Vectorizing using TF-IDF...")

    vectorizer = TfidfVectorizer()    
    vectors = vectorizer.fit_transform(texts).toarray() #Learns vocabulary accross all chunks
    #Crated vectors are numeric representation used for Similarity search

    text_embeddings = [(texts[i], vectors[i]) for i in range(len(texts))]

    class DummyEmbeddings(Embeddings):  
        def embed_documents(self, docs):
            return vectorizer.transform(docs).toarray()

        #Used to convert query -> TF-IDF vector
        def embed_query(self, query):
            return vectorizer.transform([query]).toarray()[0]

    dummy = DummyEmbeddings()

    #FAISS.from_embeddings(text_embeddings, embedding=dummy) builds an in-memory FAISS index from the vectors and attaches the dummy embedding object so later retrieval calls can call embedding.embed_query(query).
    vectordb = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=dummy
    )

    return vectordb #Returns vector store
