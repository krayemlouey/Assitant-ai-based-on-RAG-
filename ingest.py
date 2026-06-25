import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1) Charger tous les PDFs du dossier data/
data_dir = "data"
pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]

if not pdf_files:
    print(f"Erreur : Aucun fichier PDF trouve dans le dossier '{data_dir}'.")
    print("Veuillez y placer au moins un document PDF.")
    sys.exit(1)

print(f"Fichiers PDF trouves : {pdf_files}")
docs = []
for pdf_path in pdf_files:
    print(f"Chargement de : {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())

print(f"Nombre total de pages chargees : {len(docs)}")

# 2) Découpage en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
print(f"Nombre total de segments (chunks) crees : {len(chunks)}")

# 3) Embeddings model (texte → vecteurs)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4) Création index FAISS
db = FAISS.from_documents(chunks, embeddings)

# 5) Sauvegarde disque
db.save_local("vectorstore/faiss_index")

print("[OK] Index FAISS cree avec succes")
