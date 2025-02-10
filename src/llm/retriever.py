from langchain.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import os, sys

sys.path.append(".")
import constant as const

def load_db(db_name):
    print("Loading vectorstore...")

    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    dbname = const.DBNAME
    vector_store_path = f"./db/{db_name}" if db_name else const.FAISS_DB1_PATH
    print(vector_store_path)

    if os.path.exists(vector_store_path):
        vector_db1 = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Vector store not found.")
        return None

    if vector_db1:
        print(f"{dbname} Vectorstore loaded")
        return vector_db1
