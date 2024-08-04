from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

import os
import openai
import sys

sys.path.append(".")
from src.llm.retriever import load_db

"""
Configure your LLM Chain to use for this personal assistant.
"""
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def load_chain(vector_db):
    """Logic for loading the chain you want to use should go here."""

    template = """
    Give best possible answer. 
    You will be provided with a document and a corresponding question.
    If you don't know the answer, just say that you "I don't know". Don't try to make up an answer.

    
    QUESTION: {question}
    =========
    {summaries}
    =========

    ANSWER:

    """

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["summaries", "question"]
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.1)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=QA_PROMPT,
    )

    chain = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 5},
        ),
        return_source_documents=True,
    )

    return chain


def final_chain(query, db_name):
    try:
        vector_db1 = load_db(db_name=db_name)
        chain = load_chain(vector_db1)
        output = chain({"question": query}, return_only_outputs=True)
        print(output, 'output')
        return output
    except Exception as e:
        print(f"No data in DB: \n {e}")
        return { "answer": "I don't know", "sources": "" }
