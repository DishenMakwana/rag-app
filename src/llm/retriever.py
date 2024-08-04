from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os, sys

sys.path.append(".")
import constant as const

def load_db(db_name):
    print("Loading vectorstore...")
    embeddings = OpenAIEmbeddings()
    dbname = const.DBNAME
    vector_store_path = f"./db/{db_name}" if db_name else const.FAISS_DB1_PATH
    print(vector_store_path)
    if os.path.exists(vector_store_path):
        vector_db1 = FAISS.load_local(vector_store_path, embeddings)

    if vector_db1:
        print(f"{dbname} Vectorstore loaded")
        return vector_db1
