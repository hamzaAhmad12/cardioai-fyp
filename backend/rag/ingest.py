# ingest.py — Pipeline 1: Load PDFs → Chunk → Embed → Store

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os


def load_pdfs(pdf_folder: str):
    all_documents = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            print(f" Loading: {filename}")
            
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata["guideline"] = filename
                doc.metadata["source_file"] = filename
            
            all_documents.extend(documents)
            print(f"    Loaded {len(documents)} pages")
    
    print(f"\n  Total pages loaded: {len(all_documents)}")
    return all_documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"   Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks, persist_dir="./chroma_db"):
    print("\n Creating embeddings (this may take a few minutes)...")
    
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    
    print(f"  Vector store created with {len(chunks)} chunks")
    print(f"  Saved to: {persist_dir}")
    return vector_store


if __name__ == "__main__":
    print()
    print("  Medical RAG — Data Injection Pipeline")
    print()
    
    docs = load_pdfs("./data/pdfs")
    chunks = chunk_documents(docs)
    store = create_vector_store(chunks)
    
    print("\n  ✅ ")
  