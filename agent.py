"""LangChain agent wired to LM Studio and the Articles API tool."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from prompts import SYSTEM_PROMPT, ensure_contract, format_contract, parse_tool_json
from tools_articles import articles_tool, call_articles_api


def build_llm(*, temperature: float = 0.1) -> ChatOpenAI:
    """Chat model pointing at a local LM Studio OpenAI-compatible server."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.lm_studio_base_url,
        api_key=settings.lm_studio_api_key,
        model=settings.lm_studio_model,
        temperature=temperature,
    )


def build_agent_llm() -> Any:
    """LLM with the Articles tool bound for tool-calling."""
    return build_llm().bind_tools([articles_tool])


def _invoke_bound_tool(tool_call: dict[str, Any]) -> str:
    name = tool_call.get("name", "")
    args = tool_call.get("args") or {}
    if name != articles_tool.name:
        return json.dumps(
            {"ok": False, "error": f"Unknown tool: {name}", "article": None},
            ensure_ascii=False,
        )
    return call_articles_api(**args)


def run_agent(user_query: str, *, max_tool_rounds: int = 3) -> str:
    """
    Run one user query through the Articles API operator agent.

    Uses an explicit tool-calling loop so the HTTP Articles tool is actually invoked.
    """
    llm = build_agent_llm()
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    try:
        for _ in range(max_tool_rounds):
            ai_message: AIMessage = llm.invoke(messages)
            messages.append(ai_message)

            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                content = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
                return ensure_contract(content)

            for tool_call in tool_calls:
                tool_result = _invoke_bound_tool(tool_call)
                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

        # Final pass without further tools if the model keeps calling tools.
        final: AIMessage = build_llm().invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "Stop calling tools. Produce the final answer now using only the "
                        "response contract (Status/Action/Data/Errors)."
                    )
                )
            ]
        )
        content = final.content if isinstance(final.content, str) else str(final.content)
        return ensure_contract(content)
    except Exception as exc:  # noqa: BLE001
        return format_contract(
            status="error",
            action="Agent execution failed",
            data=None,
            errors=str(exc),
        )


def run_agent_with_trace(user_query: str) -> dict[str, Any]:
    """Run the agent and also return whether the Articles tool was used (via re-run trace)."""
    # Lightweight instrumentation: wrap call_articles_api usage by replaying through a traced loop.
    llm = build_agent_llm()
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
    tool_calls_made: list[dict[str, Any]] = []
    final_text = ""

    try:
        for _ in range(3):
            ai_message: AIMessage = llm.invoke(messages)
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                final_text = (
                    ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
                )
                break

            for tool_call in tool_calls:
                result = _invoke_bound_tool(tool_call)
                tool_calls_made.append(
                    {
                        "name": tool_call.get("name"),
                        "args": tool_call.get("args"),
                        "result": parse_tool_json(result),
                    }
                )
                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
        else:
            final: AIMessage = build_llm().invoke(
                messages
                + [
                    HumanMessage(
                        content=(
                            "Stop calling tools. Produce the final answer now using only the "
                            "response contract (Status/Action/Data/Errors)."
                        )
                    )
                ]
            )
            final_text = final.content if isinstance(final.content, str) else str(final.content)

        return {
            "query": user_query,
            "response": ensure_contract(final_text),
            "tool_calls": tool_calls_made,
            "used_api_tool": len(tool_calls_made) > 0,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "query": user_query,
            "response": format_contract(
                status="error",
                action="Agent execution failed",
                data=None,
                errors=str(exc),
            ),
            "tool_calls": tool_calls_made,
            "used_api_tool": len(tool_calls_made) > 0,
            "error": str(exc),
        }
