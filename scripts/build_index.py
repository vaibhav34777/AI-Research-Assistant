import os
import glob
import pickle
import faiss
from sentence_transformers import SentenceTransformer
# from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_DIR = "pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Prepare splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Load & chunk all PDFs
all_texts, metadatas = [], []
for pdf_path in glob.glob(os.path.join(PDF_DIR, "*.pdf")):
    loader = PDFPlumberLoader(pdf_path)
    docs   = loader.load()                        # returns List[Document]
    chunks = splitter.split_documents(docs)       # split into chunks

    for i, chunk in enumerate(chunks):
        all_texts.append(chunk.page_content)
        metadatas.append({
            "source": os.path.basename(pdf_path),
            "chunk":  i
        })

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
embeddings = model.encode(
    all_texts,
    batch_size=64,
    show_progress_bar=True
)
embeddings = embeddings.astype("float32")  # FAISS requires float32

# Build FAISS index
dim   = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"FAISS index has {index.ntotal} vectors")

# 6. Serialize index + data for HF Space
faiss.write_index(index, "faiss_index.idx")
with open("chunks.pkl",    "wb") as f: pickle.dump(all_texts,   f)
with open("metadatas.pkl", "wb") as f: pickle.dump(metadatas,    f)
