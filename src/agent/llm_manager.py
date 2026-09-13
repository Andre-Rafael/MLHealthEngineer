from logging import info

from dotenv import load_dotenv
from langchain.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class LlmManager:
    def __init__(self, llm_type: str, model_name: str):
        self.llm_type = llm_type
        self._model_name = model_name
        self._embedding_model = "models/gemini-embedding-001" if llm_type == "google" else "nomic-embed-text"
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if self.llm_type == "google":
            return ChatGoogleGenerativeAI(model=self._model_name)
        elif self.llm_type == "ollama":
            return ChatOllama(model=self._model_name)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def get_embeddings(self) -> Embeddings:
        info(f"Initializing embeddings for model: {self._embedding_model}")
        if self.llm_type == "google":
            return GoogleGenerativeAIEmbeddings(model=self._embedding_model)
        elif self.llm_type == "ollama":
            return OllamaEmbeddings(model=self._embedding_model)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def generate_response(self, prompt: str) -> str:
        if self.llm_type == "google":
            return self.llm.generate(prompt)
        elif self.llm_type == "ollama":
            return self.llm.embed(prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")