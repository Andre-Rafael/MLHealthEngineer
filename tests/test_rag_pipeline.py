
from pathlib import Path

from langchain_core.messages import BaseMessage
from pytest import fixture

from src.agent.llm_manager import LlmManager
from src.agent.rag_pipeline import (
    generate_response_from_model, 
    load_pdf_data, 
    process_documents_for_embedding, 
    retrieve_similar_documents
)


QUERY = "Como manter a saúde mental?"

@fixture(scope="session")
def llm_manager():
    return LlmManager(llm_type="google", model_name="gemini-2.5-flash")
    # return LlmManager(llm_type="ollama", model_name="qwen3:0.6b")

@fixture(scope="session")
def documents():
    return load_pdf_data(Path("data/docs/CartilhaSaudeMentalUFLA.pdf"))


def test_rag_pipeline(llm_manager, documents):
    vector_store = process_documents_for_embedding(documents, llm_manager)
    context_text = retrieve_similar_documents(QUERY, vector_store)
    response = generate_response_from_model(llm_manager.llm, context_text, QUERY)
    assert isinstance(response, BaseMessage)