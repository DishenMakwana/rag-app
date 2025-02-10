from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

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
    {summaries}
    =========

    ANSWER:
    """

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["summaries", "question"]
    )

    # ✅ Use LangChain's ChatOllama to properly integrate with LangChain
    llm = Ollama(model="deepseek-r1:8b")

    # Load the QA chain using the provided prompt
    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=QA_PROMPT,
    )

    # Create the retrieval chain with your vector database retriever
    chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 5},
        ),
        return_source_documents=True,
    )

    return chain
def ollama_generate(prompt):
        response = ollama.generate(model="deepseek-r1:8b", prompt=prompt)
        return response['response']  # Extracts text from Ollama's response

def final_chain(query, db_name):
    try:
        # Load your vector database using your custom load_db function
        vector_db = load_db(db_name=db_name)
        chain = load_chain(vector_db)
        output = chain({"question": query}, return_only_outputs=True)
        print(output, 'output')
        return output
    except Exception as e:
        print(f"No data in DB: \n {e}")
        return { "answer": "I don't know", "sources": "" }
