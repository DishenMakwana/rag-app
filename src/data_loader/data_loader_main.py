import shutil
import os, sys, time, timeit
import ollama

sys.path.append(".")
import constant as const
from tqdm import tqdm
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

db_paths_dict = {
    "db1": const.FAISS_DB1_PATH,
}

def read_pdf_content(file_path, pdf_file):
    if os.path.exists(file_path):
        print(f"PDF file already exists: {file_path}")
        return []

    with open(file_path, "wb") as f:
        f.write(pdf_file.read())
    print(f"\nReading PDF file: {file_path}")

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
        print("Done loading website(s).")
        return documents
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return []

def process_urls(url_passed):
    vc_st = timeit.default_timer()
    print(f"Processing URLs, time: {timeit.default_timer() - vc_st} seconds")
    return load_website_content(url_passed)

def split_text(documents):
    """
    Split documents into chunks
    """
    if not documents:
        print("No documents found to split.")
        return []

    print("\nSplitting the documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
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
        print("\nURL processing started...")
        documents.extend(process_urls(urls))
        print("URL processing finished.")

    if not documents:
        print("\nNo documents found for processing.")
        return []

    print("\nread_pre_process_data finished.")
    return split_text(documents)

def create_vector_store(docs, db_name):
    try:
        vc_st = timeit.default_timer()
        print("\ncreate_vector_store started.")

        # Use the correct HuggingFaceEmbeddings import
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        dbname = const.DBNAME
        vector_store_path = f"./db/{db_name}" if db_name else const.FAISS_DB1_PATH
        if os.path.exists(vector_store_path):
            print(f"Loading existing vector store from: {vector_store_path}")
            vectordb = FAISS.load_local(vector_store_path, embedding, allow_dangerous_deserialization=True)
            vectordb = batch(vectordb, docs, embedding, 100)
            vectordb.save_local(vector_store_path)
        else:
            print("Creating new FAISS vector store...")
            vectordb = FAISS.from_documents(documents=docs, embedding=embedding)
            vectordb.save_local(vector_store_path)

        print(f"\nThe vector store generated for {db_name}..")
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
    msg = create_vector_store(docs, db_name)

    print(f"\nVectorStore status: {msg}")
    wallclock_time_et = timeit.default_timer()
    print(f"Wallclock time: {wallclock_time_et - wallclock_time_st} seconds")
    print("Main function execution completed.")

def remove_vector_store(db_name):
    print(f"Starting deletion of vector DB: {db_name}")

    vector_store_path = f"./db/{db_name}" if db_name else const.FAISS_DB1_PATH
    data_path = "./source_data/db1"

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
