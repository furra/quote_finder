"""LangGrapg Agent diagram file"""

import json
import textwrap
from pydantic import BaseModel
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4
from typing_extensions import TypedDict


from baml_client.sync_client import b
from baml_client.types import AgentGuidelines, ExecutorStepData, Step
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.types import Command

from tools import get_author_quote, get_book_quote, get_random_quote, wikipedia_tool

load_dotenv()

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

# TODO: move this to another file

def agent_base_prompt(suffix: str) -> str:
    return f"""
        You are a helpful AI assistant, collaborating with other assistants.
        Use the provided tools to progress towards answering the question.
        If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off.
        Execute what you can to make progress.
        If you or any of the other assistants have the final answer or deliverable,
        prefix your response with `FINAL ANSWER: {{answer}}` so the team knows to stop.
        {suffix}"""

def get_agent_descriptions() -> dict[str, dict[str, str]]:
    """
    Return structured agent descriptions with capabilities and guidelines.
    Edit this function to change how the Planner/Executor reason about agents.
    """
    return {
        "Researcher": {
            "name": "Information Researcher",
            "capability": "Finds out information by searching in Wikipedia",
            "use_when": "Public information, news, current events, or external facts are needed",
            "limitations": "Cannot access private/internal company data",
            "output_format": "Raw research data from Wikipedia",
        },
        "Quoter": {
            "name": "Quote finder",
            "capability": "Gets quotes from people or books",
            "use_when": "User explicitly requests quotes, phrases, comments from books or people.",
            "limitations": "Depends on Public APIs",
            "output_format": "Quote from book or a person",
        },
        "Synthesizer": {
            "name": "Synthesizer",
            "capability": "Write summaries of findings",
            "use_when": "Previous steps are done, use this as final step - combines all previous agents' outputs",
            "limitations": "Requires data from previous steps",
            "output_format": "Coherent written summary incorporating findings",
            "position_requirement": "Should be used as final step when no more information is needed",
        },
    }



def get_agent_guidelines() -> AgentGuidelines:
    agents = get_agent_descriptions()

    agent_list = []
    guidelines = ""
    for agent, description in agents.items():
        agent_list.append(agent)
        guidelines += f"- Use `{agent}` for when {description['use_when'].lower()}\n"

    return AgentGuidelines(
        agent_list=agent_list,
        guidelines=guidelines,
    )

# llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
# llm = init_chat_model("qwen/qwen3-32b", model_provider="groq")
# reasoning_llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq")
# llm = init_chat_model("gemma2-9b-it", model_provider="groq")
llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")


class State(TypedDict):
    """State class for the graph"""

    messages: list[BaseMessage]
    user_query: str
    plan: list[Step]
    current_step: int
    agent_list: list[str]
    last_reason: str | None
    agent_query: str | None


def planner_node(state: State) -> Command[Literal['Executor']]:
    user_query = state.get("user_query", str(state["messages"][0].content))
    agent_guidelines = get_agent_guidelines()

    agent_list = agent_guidelines.agent_list
    guidelines = agent_guidelines.guidelines

    plan: list[Step] = b.GeneratePlan(
        user_query,
        " | ".join(agent_list),
        guidelines,
    )

    return Command(
        update={
            "plan": plan,
            "messages": state["messages"] + [HumanMessage(
                content=json.dumps([st.model_dump_json() for st in plan]),
                name="plan"
            )],
            "user_query": user_query,
            "current_step": 0, # if not replan else state["current_step"],
            "agent_list": ["Planner"] + agent_list,
        },
        goto="Executor"
    )


def executor_node(state: State) -> Command[Literal["Researcher", "Planner", "Quoter", "Synthesizer"]]:
    plan = state.get("plan", {})

    current_step = state.get("current_step", 0)
    current_plan_step = plan[current_step]
    current_agent = current_plan_step.agent

    agent_descriptions = get_agent_descriptions()

    agent_guidelines = (
        f"- Use Researcher when {agent_descriptions["Quoter"]["use_when"].lower()}.\n"
        f"- Use Quoter when {agent_descriptions["Quoter"]["use_when"].lower()}.\n"
        f"- Use Synthesizer when {agent_descriptions["Synthesizer"]["use_when"].lower()}"
    )

    executor_data = ExecutorStepData(
        user_query=state.get("user_query", ""),
        agent_guidelines=AgentGuidelines(
            agent_list=state.get("agent_list", []),
            guidelines=agent_guidelines
        ),
        step_number=current_step+1,
        plan_step=current_plan_step,
        messages_tail=str(state["messages"][-4:])
    )

    response = b.GenerateExecutorStepPrompt(executor_data)

    updates: dict[str, Any] = {
        "messages": state["messages"] + [HumanMessage(content=str(response), name="Executor")],
        "last_reason": response.reason,
        "agent_query": response.query,
    }

    updates["current_step"] = current_step + 1 if response.go_to == current_agent else current_step

    return Command(
        update=updates,
        goto=response.go_to.value
    ) # type: ignore


researcher_agent = create_react_agent(
    llm,
    tools=[wikipedia_tool],
    prompt=agent_base_prompt(
        "You are the Researcher. You can ONLY perform research by using the provided search tool (wikipedia_tool). "
        "Make sure the query to the search tool is precise and succint, for example, "
        "if you get asked to \"Get information on Maradona\", just search for \"Maradona\"."
        "When you have found the necessary information, end your output. "
        "Do NOT attempt to take further actions."
    )
)

def research_node(state: State) -> Command[Literal['Executor']]:
    agent_query = state.get("agent_query")
    response = researcher_agent.invoke({"messages": agent_query})

    response["messages"][-1] = HumanMessage(
        content=response["messages"][-1].content,
        name="Researcher",
    )

    return Command(
        update={
            "messages": state["messages"] + [response["messages"][-1]]
        },
        goto="Executor",
    )

quote_agent = create_react_agent(
    llm,
    tools=[get_author_quote, get_book_quote, get_random_quote],
    prompt=agent_base_prompt(
        "You give quotes. Identify which type of quote is needed and call the appropriate tool. "
        "The tools you have access are: `get_author_quote`, `get_book_quote`, and `get_random_quote`."
    )
)

def quote_node(state: State) -> Command[Literal['Executor']]:
    response = quote_agent.invoke(state)

    response["messages"][-1] = HumanMessage(
        content=response["messages"][-1].content,
        name="Quoter",
    )

    return Command(
        update={
            "messages": state["messages"] + [response["messages"][-1]],
        },
        goto='Executor',
    )


def synthesizer_node(state: State) -> Command[Literal[END]]: # type: ignore
    state_messages = state.get("messages", [])
    messages: list[str] = [
        str(msg.content)
        for msg in state_messages
        if getattr(msg, "name", None) in ("Researcher", "Quoter")
    ]

    user_query = state.get("user_query",
        state_messages[0].content
        if state_messages else ""
    )

    synthesis_prompt = """
        You are the Synthesizer. Use the context below to directly answer the user's question.
        Do not invent facts not supported by the context. If data is missing, say what's missing and,
        if helpful, offer a clearly labeled best-effort estimate with assumptions.

        Produce a concise response that fully answers the question, with the following guidance:
        - Start with the direct answer (one short paragraph or a tight bullet list).
        - If any message contains a quote, include them as a brief 'Quote: [...]' line.
        - Keep the output focused; avoid meta commentary or tool instructions.
    """

    prompt = [
        HumanMessage(
            content=(
                f"User query: {user_query}\n\n{synthesis_prompt}\n\n"
                f"Context:\n```\n{"\n===\n".join(messages)}```"
            )
        )
    ]

    response = llm.invoke(prompt)

    answer = str(response.content).strip()

    return Command(
        update={
            "final_answer": answer,
            "messages": [HumanMessage(
                content=answer,
                name="Synthesizer",
            )]
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
