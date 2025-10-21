import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Use a fixed path within the job directory for the vector store
VECTORSTORE_DIR = "vectorstore"

def build_and_save_vector_store(job_dir, logger):
    """
    Loads the analysis CSV, creates embeddings, builds a Chroma vector store,
    and persists it to disk.
    """
    csv_files = [f for f in os.listdir(job_dir) if f.endswith('_analysis_log.csv')]
    if not csv_files:
        logger.error("No analysis CSV file found in the job directory.")
        st.error("Could not find the analysis CSV file to build the RAG store.")
        return None

    csv_path = os.path.join(job_dir, csv_files[0])
    vector_store_path = os.path.join(job_dir, VECTORSTORE_DIR)
    
    logger.info(f"Loading data from {csv_path} to build vector store.")
    try:
        loader = CSVLoader(file_path=csv_path, encoding="utf-8")
        documents = loader.load()

        if not documents:
            logger.error("CSVLoader failed to load any documents from the CSV file.")
            st.error("Failed to load data from the CSV. The file might be empty.")
            return None

        logger.info(f"Loaded {len(documents)} documents. Creating embeddings with Google Generative AI...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=vector_store_path
        )
        logger.info(f"Successfully built and saved vector store to {vector_store_path}")
        return vector_store_path

    except Exception as e:
        logger.exception("An error occurred while building the vector store.")
        st.error(f"Failed to build RAG store: {e}")
        return None

@st.cache_resource
def create_rag_chain(_llm_model, job_dir, _logger):
    """
    Loads the persisted vector store and creates a RetrievalQA chain.
    """
    vector_store_path = os.path.join(job_dir, VECTORSTORE_DIR)
    _logger.info(f"Loading vector store from {vector_store_path} to create RAG chain.")
    
    if not os.path.exists(vector_store_path):
        _logger.error(f"Vector store not found at {vector_store_path}")
        st.error("Vector store not found. Please build it in Tab 2 first.")
        return None
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Get top 5 results

        prompt_template = """
        You are a helpful video analysis assistant. Use the following pieces of context, which include video transcripts, frame descriptions, and event markers, to answer the user's question.
        Provide a concise and detailed answer based ONLY on the provided context. If the context doesn't contain the answer, state that clearly. Do not make up information.
        When relevant, cite the timestamp from the context. For example: "(at 01:23)".

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        _logger.info("RAG chain created successfully.")
        return qa_chain
    except Exception as e:
        _logger.exception("Failed to create the RAG chain.")
        st.error(f"Could not create RAG chain: {e}")
        return None
