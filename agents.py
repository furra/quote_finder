from typing import TYPE_CHECKING, Any

from baml_client.sync_client import b
from baml_client.types import (
    AgentGuidelines,
    ExecutorStep,
    ExecutorStepData,
    FinalAnswer,
    Step,
)
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langsmith import traceable
from pydantic import BaseModel

from tools import get_author_quote, get_book_quote, get_random_quote, wikipedia_tool

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


load_dotenv()


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
            "use_when": "Public information, news, current events, or external facts are needed.",
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


def _get_llm() -> "BaseChatModel":
    # llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq")
    llm = init_chat_model("qwen/qwen3-32b", model_provider="groq")
    # reasoning_llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq")
    # llm = init_chat_model("gemma2-9b-it", model_provider="groq")
    # llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")
    return llm


@traceable(name="Planner", run_type="llm")
def get_plan(goal: str, agent_list: list[str], guidelines: str) -> list[Step]:
    return b.GeneratePlan(
        goal,
        " | ".join(agent_list),
        guidelines,
    )


@traceable(name="Executor", run_type="llm")
def execute_step(
    user_query: str,
    agent_list: list[str],  # should be enum type?
    step_number: int,
    plan_step: Step,
    last_messages: str,
) -> ExecutorStep:

    agent_descriptions = get_agent_descriptions()

    agent_guidelines = (
        f"- Use Researcher when {agent_descriptions["Quoter"]["use_when"].lower()}. "
        "DO NOT use this for quotes, only general information.\n"
        f"- Use Quoter when {agent_descriptions["Quoter"]["use_when"].lower()}.\n"
        f"- Use Synthesizer when {agent_descriptions["Synthesizer"]["use_when"].lower()}"
    )

    data = ExecutorStepData(
        user_query=user_query,
        agent_guidelines=AgentGuidelines(
            agent_list=agent_list,
            guidelines=agent_guidelines,
        ),
        step_number=step_number,
        plan_step=plan_step,
        last_messages=last_messages,
    )

    return b.GenerateExecutorStepPrompt(data)


class ResearcherAgent:
    agent: "CompiledStateGraph"

    def __init__(self) -> None:
        self.agent = create_react_agent(
            _get_llm(),
            tools=[wikipedia_tool],
            prompt=agent_base_prompt(
                "You are the Researcher. You can ONLY perform research by using the provided search tool (wikipedia_tool). "
                "Make sure the query to the search tool is precise and succint, for example, "
                'if you get asked to "Get information on Maradona", just search for "Maradona".'
                "When you have found the necessary information, end your output. "
                "Do NOT attempt to take further actions."
            ),
        )

    @traceable(name="Researcher", run_type="llm")
    def invoke(self, query: str) -> dict[str, Any] | Any:
        return self.agent.invoke({"messages": query})


class QuoterAgent:
    agent: "CompiledStateGraph"

    def __init__(self) -> None:
        self.agent = create_react_agent(
            _get_llm(),
            tools=[get_author_quote, get_book_quote, get_random_quote],
            prompt=agent_base_prompt(
                "You give quotes. Identify which type of quote is needed and call the appropriate tool. "
                "The tools you have access are: `get_author_quote`, `get_book_quote`, and `get_random_quote`."
                "Call the chosen tool only once; if it fails, say so and stop. Do NOT attempt to take further actions."
            ),
        )

    @traceable(name="Quoter", run_type="llm")
    def invoke(self, query: str) -> dict[str, Any] | Any:
        return self.agent.invoke({"messages": query})


@traceable(name="Synthesizer", run_type="llm")
def generate_answer(user_query: str, messages: list[str]) -> FinalAnswer:
    return b.GenerateAnswer(
        user_query,
        messages,
    )
