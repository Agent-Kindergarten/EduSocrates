import os
import tempfile
import streamlit as st
from typing import List
from dotenv import load_dotenv
from deprecated import deprecated
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms.openai import OpenAI
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationKGMemory, ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema.vectorstore import VectorStore
from langchain.schema.retriever import BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS
from socrates_chain import SocratesChain
from sidebar_modules import *
from main_page_modules import *


def get_docs_from_pdfs() -> List[Document]:
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in st.session_state["pdf_docs"]:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    return docs


def get_document_chunks(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(docs)
    return document_chunks


def get_vectorstore(document_chunks: List[Document]) -> VectorStore:
    """
    This function first creates an embedding object,
    then returns a vectorstore from a list of documents.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embeddings)
    return vectorstore


def get_retriever(vectorstore: VectorStore) -> BaseRetriever:
    retriever = vectorstore.as_retriever()
    return retriever


def get_llm() -> BaseLLM:
    return ChatOpenAI(
        model=st.session_state["model_name"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
    )


def get_prompt_template():
    return PromptTemplate.from_template("{question}")


def get_chain_by_pdfs() -> Chain | None:
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("❌ Please set your OpenAI API key in the sidebar.")
        return

    # get documents
    documents = get_docs_from_pdfs()
    # split documents into chunks
    document_chunks = get_document_chunks(documents)
    vectorstore = get_vectorstore(document_chunks)
    retriever = get_retriever(vectorstore)
    llm = get_llm()
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=st.session_state["langchain_messages"],
        return_messages=True,
    )
    prompt_template = get_prompt_template()
    # Setup chain
    chain = ConversationalRetrievalChain.from_llm(
        retriever=retriever,
        llm=llm,
        condense_question_prompt=prompt_template,
        memory=memory,
        verbose=True,
    )
    return chain

