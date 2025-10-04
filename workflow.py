"""LangGrapg Agent diagram file"""

import json
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4
from typing_extensions import TypedDict

from dotenv import load_dotenv

from baml_client.types import Step
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langsmith import traceable

from agents import (
    QuoterAgent,
    ResearcherAgent,
    execute_step,
    generate_answer,
    get_agent_guidelines,
    get_plan,
)

load_dotenv()

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph


class State(TypedDict):
    """State class for the graph"""

    messages: list[BaseMessage]
    user_query: str
    plan: list[Step]
    current_step: int
    agent_list: list[str]
    last_reason: str | None
    agent_query: str | None


@traceable(name="Planner")
def planner_node(state: State) -> Command[Literal["Executor"]]:
    user_query = state.get("user_query", str(state["messages"][0].content))

    agent_guidelines = get_agent_guidelines()

    agent_list = agent_guidelines.agent_list
    guidelines = agent_guidelines.guidelines

    plan = get_plan(
        user_query,
        agent_list,
        guidelines,
    )

    return Command(
        update={
            "plan": plan,
            "messages": state["messages"]
            + [
                HumanMessage(
                    content=json.dumps([st.model_dump_json() for st in plan]),
                    name="plan",
                )
            ],
            "user_query": user_query,
            "current_step": 0,
            "agent_list": ["Planner"] + agent_list,
        },
        goto="Executor",
    )


@traceable(name="Executor")
def executor_node(
    state: State,
) -> Command[Literal["Researcher", "Quoter", "Synthesizer"]]:
    plan = state.get("plan", {})

    current_step = state.get("current_step", 0)
    current_agent = plan[current_step].agent

    response = execute_step(
        user_query=state.get("user_query", ""),
        agent_list=state.get("agent_list", []),
        step_number=current_step + 1,
        plan_step=plan[current_step],
        last_messages=str(state["messages"][-4:]),
    )

    updates: dict[str, Any] = {
        "messages": state["messages"]
        + [HumanMessage(content=str(response), name="Executor")],
        "last_reason": response.reason,
        "agent_query": response.query,
    }

    updates["current_step"] = (
        current_step + 1 if response.go_to == current_agent else current_step
    )

    return Command(update=updates, goto=response.go_to.value)


researcher_agent = ResearcherAgent()


@traceable(name="Researcher")
def research_node(state: State) -> Command[Literal["Executor"]]:
    agent_query = state.get("agent_query")
    response = researcher_agent.invoke(agent_query)

    response["messages"][-1] = HumanMessage(
        content=response["messages"][-1].content,
        name="Researcher",
    )

    return Command(
        update={"messages": state["messages"] + [response["messages"][-1]]},
        goto="Executor",
    )


quoter_agent = QuoterAgent()


@traceable(name="Quoter")
def quote_node(state: State) -> Command[Literal["Executor"]]:
    agent_query = state.get("agent_query")
    response = quoter_agent.invoke(agent_query)

    response["messages"][-1] = HumanMessage(
        content=response["messages"][-1].content,
        name="Quoter",
    )

    return Command(
        update={
            "messages": state["messages"] + [response["messages"][-1]],
        },
        goto="Executor",
    )


@traceable(name="Synthesizer")
def synthesizer_node(state: State) -> Command[Literal[END]]:  # type: ignore  # Pylint doesn't like END
    state_messages = state.get("messages", [])
    messages: list[str] = [
        f"{getattr(msg, "name")}: {msg.content}"
        for msg in state_messages
        if getattr(msg, "name", None) in ("Researcher", "Quoter")
    ]

    user_query = state.get(
        "user_query", state_messages[0].content if state_messages else ""
    )

    response = generate_answer(user_query, messages)

    answer = response.text

    return Command(
        update={
            "final_answer": answer,
            "messages": [
                HumanMessage(
                    content=answer,
                    name="Synthesizer",
                )
            ],
        },
        goto=END,
    )


def save_diagram_image(workflow: "CompiledStateGraph", image_name: str = "diagram.png"):
    """Saves agent's workflow diagram in a png image"""
    image_bytes = workflow.get_graph().draw_mermaid_png()

    with open(image_name, "wb") as file:
        file.write(image_bytes)


def initialize_graph():
    """Creates graph workflow"""
    memory = InMemorySaver()
    graph = StateGraph(State)

    graph.add_node("Planner", planner_node)
    graph.add_node("Executor", executor_node)
    graph.add_node("Researcher", research_node)
    graph.add_node("Quoter", quote_node)
    graph.add_node("Synthesizer", synthesizer_node)

    graph.add_edge(START, "Planner")

    return graph.compile(checkpointer=memory)


@traceable(name="AgentGraph")
def stream(
    graph: "CompiledStateGraph",
    user_input: str,
    thread_id: str | None = None,
) -> list[str]:
    """Streams user input to the graph and prints every event step"""
    if thread_id is None:
        thread_id = str(uuid4())
    config: "RunnableConfig" = {"configurable": {"thread_id": thread_id}}
    return [
        event["messages"][-1].content
        for event in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )
    ]
