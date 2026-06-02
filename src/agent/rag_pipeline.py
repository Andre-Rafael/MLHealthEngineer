import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chat model used to generate the final answer from the retrieved context.
MODEL_NAME = "gemini-2.5-flash"

# Load environment variables (e.g. GOOGLE_API_KEY) from a local .env file.
load_dotenv()


def load_pdf_data(pdf_path: Path) -> List[Document]:
    """Load a PDF from disk and return its pages as a list of Documents."""
    logger.info("Loading PDF data from %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    logger.info("Loaded %d document(s) from %s", len(documents), pdf_path)
    return documents


def process_documents_for_embedding(documents: List[Document]) -> Chroma:
    logger.info("Splitting documents into chunks (chunk_size=500, chunk_overlap=50)")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    logger.info("Generated %d chunks from %d document(s)", len(chunks), len(documents))

    EMBEDDING_MODEL = "models/gemini-embedding-001"
    logger.info(f"Initializing embeddings model: {EMBEDDING_MODEL}")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    logger.info("Building Chroma vector store (persist_directory='chroma_db')")
    chroma_vectore_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="chroma_db",
    )
    logger.info("Chroma vector store ready")
    return chroma_vectore_store


def retrieve_similar_documents(query: str, chroma_vectore_store: Chroma) -> str:
    logger.info("Running similarity search for query: %s", query)
    docs_chroma = chroma_vectore_store.similarity_search_with_score(query, k=5)
    logger.info("Retrieved %d chunk(s) from the vector store", len(docs_chroma))

    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])
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


def generate_response_from_model(MODEL_NAME) -> AIMessage:
    prompt_template = get_prompt_template()
    prompt = prompt_template.format(context=context_text, question=query)

    logger.info("Invoking language model: %s", MODEL_NAME)
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0)
    response_text = model.invoke(prompt)
    logger.info("Received response from language model")

    return response_text


if __name__ == "__main__":
    logger = logging.getLogger("RAG Pipeline")
    logger.info("Starting RAG pipeline")
    query = input("Bem vindo ao pipeline RAG, digite sua pergunta: ")
    documents = load_pdf_data(Path("data/docs/CartilhaSaudeMentalUFLA.pdf"))
    chroma_vectore_store = process_documents_for_embedding(documents)
    context_text = retrieve_similar_documents(query, chroma_vectore_store)
    response = generate_response_from_model(MODEL_NAME)
    print("Resposta:", response.content)
