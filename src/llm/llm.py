from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os
import sys
sys.path.append(".")

from src.llm.retriever import load_db
from dotenv import load_dotenv
load_dotenv()

# Import the ollama package (make sure it's installed)
import ollama

def load_chain(vector_db):
    """
    Load your locally running Ollama model instead of OpenAI.
    Assumes that the `deepseek-r1:8b` model is available locally via Ollama.
    """
    template = """
    Give the best possible answer. 
    You will be provided with a document and a corresponding question.
    If you don't know the answer, just say "I don't know". Don't try to make up an answer.

    QUESTION: {question}
    =========
    {context}
    =========

    ANSWER:
    """

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    llm = OllamaLLM(model="deepseek-r1:8b")

    # Load the QA chain using the provided prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 5},
        ),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    return qa_chain


def ollama_generate(prompt):
        response = ollama.generate(model="deepseek-r1:8b", prompt=prompt)
        return response['response']  # Extracts text from Ollama's response

def final_chain(query, db_name):
    try:
        # Load your vector database using your custom load_db function
        vector_db = load_db(db_name=db_name)
        chain = load_chain(vector_db)
        output = chain.invoke({"query": query}, return_sources=True)
        print(output, 'output')
        return output
    except Exception as e:
        print(f"No data in DB: \n {e}")
        return { "answer": "I don't know", "sources": "" }
