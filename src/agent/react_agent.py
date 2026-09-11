from logging import warning
from typing import List
from dotenv import load_dotenv

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import load_pdf_data, search_info_in_gov_br, search_news_in_g1

load_dotenv()
MODEL_NAME = "gemini-2.5-flash"


def generate_react_prompt() -> PromptTemplate:
    REACT_PROMPT = PromptTemplate.from_template("""Você é um assistente especializado.
        Use as ferramentas disponíveis para responder perguntas.
        Ferramentas disponíveis:
        {tools}
        Use o formato:
        Thought: pensar sobre o que fazer
        Action: ação a tomar, deve ser uma das [{tool_names}]
        Action Input: input para a ferramenta
        Observation: resultado da ferramenta
        ... (repita Thought/Action/Observation quantas vezes necessário)
        Thought: Agora sei a resposta final
        Final Answer: resposta para o usuário
        Pergunta: {input}
        {agent_scratchpad}""")
    
    return REACT_PROMPT



def generate_tools() -> List[Tool]:
    return [
        Tool(
            name="load_pdf_data",
            description="Carrega o conteúdo de um documento PDF sobre saúde mental para o agente.",
            func=load_pdf_data,
            kwargs={"pdf_path": "data/docs/CartilhaSaudeMentalUFLA.pdf"},
        ),
        Tool(
            name="search_news_in_g1",
            description="""
                Procura por notícias recentes sobre saúde mental no site G1 para o agente.
                A consulta deve ser feita em português e use palavras-chave relevantes para obter os melhores resultados.
            """,
            func=search_news_in_g1,
        ),
        Tool(
            name="search_info_in_gov_br",
            description="""
                Procura por informações sobre saúde mental no site do governo brasileiro para o agente.
                A consulta deve ser feita em português e use palavras-chave relevantes para obter os melhores resultados.
            """,
            func=search_info_in_gov_br,
        ),
    ]


def create_datathon_agent(
    tools: list[Tool],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> AgentExecutor:
    """Cria agente ReAct para o Datathon.

    Args:
        tools: Lista de ferramentas (≥ 3 obrigatório).
        model_name: Modelo LLM a utilizar.
        temperature: Temperatura de geração.

    Returns:
        AgentExecutor configurado.
    """
    react_prompt = generate_react_prompt()
    if len(tools) < 3:
        warning("Datathon exige ≥ 3 tools. Fornecidas: %d", len(tools))

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )


if __name__ == "__main__":
    tools = generate_tools()
    agent_executor = create_datathon_agent(tools, model_name=MODEL_NAME, temperature=0.0)
    response = agent_executor.invoke(
        {"input": "Resume a nova legislação sobre saude mental no ambiente de trabalho no Brasil."}
    )
    print(response)