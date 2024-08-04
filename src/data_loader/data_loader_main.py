import shutil
import os, sys, time, timeit
import openai

sys.path.append(".")
import constant as const
from tqdm import tqdm
from langchain.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

db_paths_dict = {
    "db1": const.FAISS_DB1_PATH,
}

def read_pdf_content(file_path, pdf_file):
    if os.path.exists(file_path):
        print(f"PDF file already exists: {file_path}")
        return []

    with open(file_path, "wb") as f:
        f.write(pdf_file.read())
    print(f"\Reading pdf file: {file_path}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

def save_pdf_content(file_path, pdf_files):
    if os.path.exists(file_path):
        print(f"PDF file already exists: {file_path}")
        return []
    
    with pdf_files.file as buffer:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(buffer, f)

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

def load_website_content(url):
    """
    Load website content into Documents
    """

    try:
        print("\nLoading website(s) into Documents...")
        documents = WebBaseLoader(url).load()
        print("\n\n")
        print("\nDone loading website(s).\n", "-" * 50)
        return documents
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return []


def process_urls(url_passed):
    vc_st = timeit.default_timer()

    print(f"process_urls time: {timeit.default_timer() - vc_st} seconds")
    # Step 2: Load the website content into Documents
    return load_website_content(url_passed)

def split_text(documents):
    """
    Split documents into chunks
    """
    print("\nSplitting the documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print("\nDone splitting documents.\n", "-" * 50)
    return chunks


def batch(vectordb, docs, embedding, size):
    print("Batch processing started...")
    print(f"Total number of documents: {len(docs)}")
    
    for i in range(0, len(docs), size):
        print(f"Processing Batch No: {i}")
        current_batch = docs[i : i + size]
        time.sleep(1)
        vectordb2 = FAISS.from_documents(documents=current_batch, embedding=embedding)
        vectordb.merge_from(vectordb2)
    
    print("Batch processing completed.")
    return vectordb


def read_pre_process_data(pdf_files, urls):
    documents = []
    temp_path = "./source_data/db1"
    os.makedirs(temp_path, exist_ok=True)

    if pdf_files:
        if hasattr(pdf_files, 'name'):
            print("\nReading uploaded PDF file...")
            file_path = os.path.join(temp_path, pdf_files.name)
            documents = read_pdf_content(file_path, pdf_files)
        else:
            print("\nReading existing PDF file...")
            file_path = os.path.join(temp_path, pdf_files.filename)
            documents = save_pdf_content(file_path, pdf_files)    

    if urls:
        print("\nurl processing started...")
        documents.extend(process_urls(urls))
        print("url processing finish...")

    print("\nread_pre_process_data finished..")
    return split_text(documents)


def create_vector_store(docs, db_name):
    try:
        vc_st = timeit.default_timer()
        print("\ncreate_vector_store started.")

        embedding = OpenAIEmbeddings()

        dbname = const.DBNAME
        vector_store_path = f"./db/{db_name}" if db_name else const.FAISS_DB1_PATH
        if os.path.exists(vector_store_path):
            print(f"Loading existing vector store from: {vector_store_path}")
            vectordb = FAISS.load_local(vector_store_path, embedding)
            vectordb = batch(vectordb, docs, embedding, 100)
            vectordb.save_local(vector_store_path)
        else:
            print("Faiss VectorDB used to Load the vectorstore.")
            vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
            vectordb.save_local(vector_store_path)

        print(f"\nThe vector store generated for {dbname}..")
        vc_et = timeit.default_timer()
        print(f"Vector store generation time: {vc_et - vc_st} seconds")
        print("\ncreate_vector_store finished.")
        return "success"

    except Exception as e:
        print(f"Error in create_vector_store: \n {e}")
        return "failure"


def process_vector_store(pdf_files, urls, db_name):
    print("Main function execution started..")
    wallclock_time_st = timeit.default_timer()

    docs = read_pre_process_data(pdf_files, urls)

    # step2: create a vectorstore
    msg = create_vector_store(docs, db_name)
    print(f"\nVectorStore status: {msg}")

    wallclock_time_et = timeit.default_timer()
    print(f"Wallclock time: {wallclock_time_et - wallclock_time_st} seconds")
    print("Main function execution completed..")

def remove_vector_store(db_name):
    print(db_name, 'db_name')
    print("Starting deleting vector DB")
    vector_store_path = f"{db_name}" if db_name else const.FAISS_DB1_PATH
    data_path = f"./source_data/db1"
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        os.makedirs(data_path)

    if os.path.exists(vector_store_path):
        try:
            shutil.rmtree(vector_store_path)

            print("Vector DB deleted successfully")
        except OSError as err:
            print(f"Error removing vector store file: {err}")
    else:
        print("Vector DB file does not exist")