"""LangGrapg Agent diagram file"""

import textwrap
from typing import TYPE_CHECKING
from uuid import uuid4
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tools import get_author_quote, get_book_quote, get_random_quote

load_dotenv()

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

system_prompt = SystemMessage(
    content=(
        "You are a helpful assistant that finds quotes. "
        "You have access to these tools: `get_author_quote`, `get_book_quote`, and `get_random_quote`. "
        "You may only call a tool **once per author or book**. "
        "If you already called a tool and received a quote from that author/book, do not call it again. "
        "Return the quote text directly to the user and stop. "
        "Do not generate additional tool calls unless a new user input is received."
        "If `get_author_quote` or `get_book_quote` fails due to a timeout or returns a message like "
        "'Couldn't find any quote from ...', then call `get_random_quote` instead."
    )
)

# llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
llm = init_chat_model("qwen/qwen3-32b", model_provider="groq")
# llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq")
# llm = init_chat_model("gemma2-9b-it", model_provider="groq")
# llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")

tools = [get_author_quote, get_book_quote, get_random_quote]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    """State class for the graph"""

    messages: list[BaseMessage]


def should_continue(state: State) -> str:
    """Function to determine if chatbot finishes"""
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        print(
            "================================= Reasoning ===================================="
        )
        print(textwrap.fill(last_message.additional_kwargs["reasoning_content"], width=150))
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


def chatbot(state: State) -> State:
    """Chatbot interface"""
    last_message = state["messages"][-1]

    if (
        isinstance(last_message, ToolMessage)
        and "Couldn't find" not in last_message.content
    ):
        return state

    return {
        "messages": [
            system_prompt,
            llm_with_tools.invoke([system_prompt] + state["messages"]),
        ]
    }


def save_diagram_image(workflow: "CompiledStateGraph", image_name: str = "diagram.png"):
    """Saves agent's workflow diagram in a png image"""
    image_bytes = workflow.get_graph().draw_mermaid_png()

    with open(image_name, "wb") as file:
        file.write(image_bytes)


def initialize_graph():
    """Creates graph workflow"""
    memory = InMemorySaver()
    graph = StateGraph(State)

    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", should_continue, ["tools", END])
    graph.add_edge("tools", "chatbot")
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
