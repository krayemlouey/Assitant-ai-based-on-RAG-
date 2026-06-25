from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA

# 1) Charger embeddings (même modèle que ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2) Charger index FAISS déjà stocké
db = FAISS.load_local(
    "vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 3) LLaMA (LLM local via Ollama)
llm = ChatOllama(model="llama3.2")

# 4) RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

def ask_question(question):
    result = qa.invoke({"query": question})
    return result["result"]
