"""LangChain tool that calls the public Articles API."""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from articles_api import create_article, get_article, update_article


class ArticlesToolInput(BaseModel):
    """Input schema for the Articles API tool."""

    operation: Literal["create", "get", "update"] = Field(
        description="API operation: create a new article, get by id, or update an existing article."
    )
    article_id: Optional[int] = Field(
        default=None,
        description="Required for get and update. Numeric article id.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Article title. Required for create; optional for update.",
    )
    body: Optional[str] = Field(
        default=None,
        description=(
            "Article body/content. For create, prefer providing body; "
            "if omitted, a short body is derived from the title. Optional for update."
        ),
    )
    user_id: Optional[int] = Field(
        default=1,
        description="Author user id. Defaults to 1 for create; optional for update.",
    )


def call_articles_api(
    operation: Literal["create", "get", "update"],
    article_id: int | None = None,
    title: str | None = None,
    body: str | None = None,
    user_id: int | None = 1,
) -> str:
    """
    Explicit HTTP wrapper used by the LangChain tool.

    Returns a JSON string (structured dict → string) for the agent.
    """
    result: dict[str, Any]
    try:
        if operation == "create":
            if not title:
                raise ValueError("create requires title.")
            # Local models often omit body; derive a short default from the title.
            resolved_body = (body or "").strip() or f"{title.strip()}."
            result = create_article(
                title=title.strip(),
                body=resolved_body,
                user_id=user_id or 1,
            )
        elif operation == "get":
            if article_id is None:
                raise ValueError("get requires article_id.")
            result = get_article(article_id=article_id)
        elif operation == "update":
            if article_id is None:
                raise ValueError("update requires article_id.")
            result = update_article(
                article_id=article_id,
                title=title,
                body=body,
                user_id=user_id,
            )
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        return json.dumps({"ok": True, "error": None, **result}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 — surface all API/tool errors to the agent
        return json.dumps(
            {
                "ok": False,
                "error": str(exc),
                "operation": operation,
                "article": None,
            },
            ensure_ascii=False,
        )


articles_tool = StructuredTool.from_function(
    func=call_articles_api,
    name="articles_api",
    description=(
        "Call the public Articles API. "
        "Supported operations: create (title required; body optional — "
        "defaults from title), get (article_id), "
        "update (article_id + optional title/body/user_id). "
        "Returns a JSON string with ok, operation, article, and error fields."
    ),
    args_schema=ArticlesToolInput,
)
