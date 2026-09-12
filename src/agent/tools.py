from pathlib import Path

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from requests import get
from logging import info



def load_pdf_data(pdf_path: Path) -> str:
    """Load a PDF from disk and return its pages as a single string."""
    info("Loading PDF data from %s", pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    info("Loaded %d document(s) from %s", len(documents), pdf_path)
    return "\n\n".join([doc.page_content for doc in documents])


def search_news_in_g1(query: str) -> str:
    info("Searching for news on G1 with query: %s", query)
    url = f"https://g1.globo.com/busca/?q={"+".join(query.split())}"
    response = get(url, timeout=30)

    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def search_info_in_gov_br(query: str) -> str:
    info("Searching for information on gov.br with query: %s", query)
    url = f"https://www.gov.br/saude/pt-br/search?origem=form&SearchableText={"+".join(query.split())}"
    response = get(url, timeout=30)

    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()