from logging import basicConfig, getLogger, INFO, info
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "gemini-2.5-flash"

load_dotenv()

basicConfig(level=INFO)
logger = getLogger("RAG Pipeline")


def load_pdf_data(pdf_path: Path) -> List[Document]:
    """Load a PDF from disk and return its pages as a list of Document objects."""
    info("Loading PDF data from %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    info("Loaded %d document(s) from %s", len(documents), pdf_path)
    return documents

def process_documents_for_embedding(documents: List[Document]) -> Chroma:
    info("Splitting documents into chunks (chunk_size=900, chunk_overlap=50)")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    info("Generated %d chunks from %d document(s)", len(chunks), len(documents))

    EMBEDDING_MODEL = "models/gemini-embedding-001"
    info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    info("Building Chroma vector store (persist_directory='chroma_db')")
    if not Path("chroma_db").exists():
        info("No existing Chroma vector store found, creating a new one from chunks")
        chroma_vector_store = Chroma.from_documents(
            chunks,
            persist_directory="chroma_db",
            embedding=embeddings,
        )
    else:
        chroma_vector_store = Chroma.from_documents(
            chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",
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


def generate_response_from_model(model_name: str, context_text: str, query: str) -> BaseMessage:
    prompt_template = get_prompt_template()
    prompt = prompt_template.format(context=context_text, question=query)

    info("Invoking language model: %s", model_name)
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    response_text = model.invoke(prompt)
    info("Received response from language model")

    return response_text


if __name__ == "__main__":
    info("Starting RAG pipeline")
    query = input("Bem vindo ao pipeline RAG, digite sua pergunta: ")
    documents = load_pdf_data(Path("data/docs/CartilhaSaudeMentalUFLA.pdf"))
    vector_store = process_documents_for_embedding(documents)
    context_text = retrieve_similar_documents(query, vector_store)
    response = generate_response_from_model(MODEL_NAME, context_text, query)
    print("Resposta:", response.content)
