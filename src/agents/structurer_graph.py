"""LangGraph orchestrator for the Autocall Structurer agent."""

from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import AsyncOpenAI

from src.models.schemas import AutocallSpecs, MarketDataInput, PricingResult, StructurerState
from src.tools.market_data import get_market_data
from src.tools.pricer import solve_target_coupon
from src.tools.term_sheet_gen import generate_term_sheet_markdown


BANK_TARGET_FAIR_VALUE = 98.5


def check_node_status(state: StructurerState) -> Literal["continue", "abort"]:
    """Conditional router stopping graph execution if an error occurred."""
    if state.get("status") == "failed" or state.get("error_message"):
        return "abort"
    return "continue"


async def parse_request_node(state: StructurerState) -> StructurerState:
    client = AsyncOpenAI()
    client_request = state.get("client_request", "")
    if not client_request:
        state["error_message"] = "No client request provided"
        state["status"] = "failed"
        return state

    try:
        completion = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured autocall parameters from the client request. Translate tickers accurately (e.g., Euro Stoxx 50 -> ^STOXX50E).",
                },
                {"role": "user", "content": client_request},
            ],
            response_format=AutocallSpecs,
        )
        parsed_specs: AutocallSpecs = completion.choices[0].message.parsed
        state["specs"] = parsed_specs.model_dump()
        state["status"] = "specs_parsed"
    except Exception as exc:
        state["error_message"] = f"Failed to parse request: {exc}"
        state["status"] = "failed"

    return state


async def fetch_data_node(state: StructurerState) -> StructurerState:
    try:
        specs_dict = state.get("specs", {})
        ticker = specs_dict.get("underlying", "")
        market_data = await get_market_data(ticker)
        state["market_data"] = market_data.model_dump()
        state["status"] = "market_data_fetched"
    except Exception as exc:
        state["error_message"] = f"Failed to fetch market data: {exc}"
        state["status"] = "failed"

    return state


def run_pricing_node(state: StructurerState) -> StructurerState:
    try:
        specs = AutocallSpecs(**state["specs"])
        market_data = MarketDataInput(**state["market_data"])
        pricing_result = solve_target_coupon(specs, market_data, target_fair_value=BANK_TARGET_FAIR_VALUE)
        state["pricing_result"] = pricing_result.model_dump()
        state["status"] = "pricing_complete"
    except Exception as exc:
        state["error_message"] = f"Failed to run pricing: {exc}"
        state["status"] = "failed"

    return state


def approval_node(state: StructurerState) -> StructurerState:
    interrupt("Validation manuelle requise pour le pricing")
    return state


async def generate_termsheet_node(state: StructurerState) -> StructurerState:
    try:
        specs = AutocallSpecs(**state["specs"])
        market_data = MarketDataInput(**state["market_data"])
        pricing_result = PricingResult(**state["pricing_result"])

        term_sheet_md = await generate_term_sheet_markdown(specs, market_data, pricing_result)
        state["term_sheet_md"] = term_sheet_md
        state["status"] = "COMPLETED"
    except Exception as exc:
        state["error_message"] = f"Failed to generate term sheet: {exc}"
        state["status"] = "failed"

    return state


def build_graph():
    graph = StateGraph(StructurerState)
    memory = MemorySaver()

    graph.add_node("parse_request", parse_request_node)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("run_pricing", run_pricing_node)
    graph.add_node("approval", approval_node)
    graph.add_node("generate_termsheet", generate_termsheet_node)

    graph.add_edge(START, "parse_request")

    graph.add_conditional_edges(
        "parse_request",
        check_node_status,
        {"continue": "fetch_data", "abort": END},
    )
    graph.add_conditional_edges(
        "fetch_data",
        check_node_status,
        {"continue": "run_pricing", "abort": END},
    )
    graph.add_conditional_edges(
        "run_pricing",
        check_node_status,
        {"continue": "approval", "abort": END},
    )

    graph.add_edge("approval", "generate_termsheet")
    graph.add_edge("generate_termsheet", END)

    return graph.compile(checkpointer=memory)