"""HTTP client for the public Articles API (JSONPlaceholder /posts)."""

from __future__ import annotations

from typing import Any

import httpx

from config import get_settings


def _client() -> httpx.Client:
    settings = get_settings()
    return httpx.Client(
        base_url=settings.articles_api_base_url,
        timeout=30.0,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )


def create_article(
    title: str,
    body: str,
    user_id: int = 1,
) -> dict[str, Any]:
    """POST /posts — create an article."""
    payload = {"title": title, "body": body, "userId": user_id}
    with _client() as client:
        response = client.post("/posts", json=payload)
        response.raise_for_status()
        return {"operation": "create", "status_code": response.status_code, "article": response.json()}


def get_article(article_id: int) -> dict[str, Any]:
    """GET /posts/{id} — fetch an article by id."""
    with _client() as client:
        response = client.get(f"/posts/{article_id}")
        response.raise_for_status()
        return {"operation": "get", "status_code": response.status_code, "article": response.json()}


def update_article(
    article_id: int,
    title: str | None = None,
    body: str | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    """PATCH /posts/{id} — update an article."""
    payload: dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    if body is not None:
        payload["body"] = body
    if user_id is not None:
        payload["userId"] = user_id
    if not payload:
        raise ValueError("At least one of title, body, or user_id must be provided for update.")

    with _client() as client:
        response = client.patch(f"/posts/{article_id}", json=payload)
        response.raise_for_status()
        return {"operation": "update", "status_code": response.status_code, "article": response.json()}
