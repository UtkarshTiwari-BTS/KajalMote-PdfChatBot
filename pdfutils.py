import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. EXTRACT TEXT FROM PDF
def extract_pdf_text(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# 2. CHUNK TEXT
def chunk_text(text, chunk_size=2500, chunk_overlap=400):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)
