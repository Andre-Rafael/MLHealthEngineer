from conftest import llm_manager
from src.agent.react_agent import create_datathon_agent, generate_tools

def test_create_datathon_agent(llm_manager):
    tools = generate_tools()
    agent_executor = create_datathon_agent(tools=tools, llm=llm_manager.llm)

    assert agent_executor is not None
    assert len(agent_executor.tools) >= 3


def test_agent_executor_invocation(llm_manager):
    tools = generate_tools()
    agent_executor = create_datathon_agent(tools=tools, llm=llm_manager.llm)

    response = agent_executor.invoke(
        {"input": "Resume a nova legislação sobre saude mental no ambiente de trabalho no Brasil."}
    )

    assert "output" in response
    assert isinstance(response["output"], str)