from logging import basicConfig, getLogger, INFO, info
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent.llm_manager import LlmManager

load_dotenv()

basicConfig(level=INFO)
logger = getLogger("RAG Pipeline")


def load_pdf_data(pdf_path: Path) -> List[Document]:
    """Load a PDF from disk and return its pages as a list of Document objects."""
    info("Loading PDF data from %s", pdf_path)
    reader = PdfReader(pdf_path)
    documents = [Document(page_content=page.extract_text()) for page in reader.pages]
    info("Loaded %d document(s) from %s", len(documents), pdf_path)
    return documents

def process_documents_for_embedding(documents: List[Document], llm_manager: LlmManager) -> Chroma:
    info("Splitting documents into chunks (chunk_size=900, chunk_overlap=50)")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    info("Generated %d chunks from %d document(s)", len(chunks), len(documents))

    # EMBEDDING_MODEL = "models/gemini-embedding-001"
    EMBEDDING_MODEL = llm_manager._embedding_model
    info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
    embeddings = llm_manager.get_embeddings()

    info("Building Chroma vector store (persist_directory='chroma_db')")
    chroma_vector_store = Chroma.from_documents(
        chunks,
        persist_directory="chroma_db",
        embedding=embeddings,
    )
    info("Chroma vector store ready")
    return chroma_vector_store


def retrieve_similar_documents(query: str, vector_store: Chroma) -> str:
    info("Running similarity search for query: %s", query)
    docs_chroma = vector_store.similarity_search(query, k=5)
    info("Retrieved %d chunk(s) from the vector store", len(docs_chroma))
    context_text = "\n\n".join([doc.page_content for doc in docs_chroma])
    return context_text


def get_prompt_template():
    PROMPT_TEMPLATE = """
        Answer the question based only on the following context:
        {context}
        Answer the question based on the above context: {question}.
        Provide a detailed answer.
        Don't justify your answers.
        Don't give information not mentioned in the CONTEXT INFORMATION.
        Do not say "according to the context" or "mentioned in the context" or similar.
    """
    return ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def generate_response_from_model(generative_model: Embeddings, context_text: str, query: str) -> BaseMessage:
    prompt_template = get_prompt_template()
    prompt = prompt_template.format(context=context_text, question=query)

    # info("Invoking language model: %s", model_name)
    response_text = generative_model.invoke(prompt)
    info("Received response from language model")

    return response_text
