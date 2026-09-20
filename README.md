# LangChain Articles API Agent

A small LangChain agent that acts as an **Articles API operator**. It talks to a local LLM through **LM Studio** (OpenAI-compatible API) and performs create / get / update against a public test Articles API.

## Domain: Articles

Public test API: [JSONPlaceholder](https://jsonplaceholder.typicode.com) (`/posts` treated as articles).

| Operation | HTTP | Endpoint |
|-----------|------|----------|
| create | `POST` | `/posts` |
| get | `GET` | `/posts/{id}` |
| update | `PATCH` | `/posts/{id}` |

Base URL is configurable via `ARTICLES_API_BASE_URL` (default: `https://jsonplaceholder.typicode.com`).

## Configuration (no secrets in the repo)

1. Copy the example env file:

```bash
cp .env.example .env
```

2. Start [LM Studio](https://lmstudio.ai/), load a chat model, and enable the local server (default `http://127.0.0.1:1234`).

3. Adjust `.env` if needed:

| Variable | Purpose | Example |
|----------|---------|---------|
| `LM_STUDIO_BASE_URL` | OpenAI-compatible base URL | `http://127.0.0.1:1234/v1` |
| `LM_STUDIO_API_KEY` | Placeholder key required by the OpenAI client (LM Studio does not validate it) | `lm-studio` |
| `LM_STUDIO_MODEL` | Model id as shown in LM Studio | `local-model` or the loaded model name |
| `ARTICLES_API_BASE_URL` | Articles API root | `https://jsonplaceholder.typicode.com` |

Do **not** commit `.env`. Only `.env.example` is tracked.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run from CLI

```bash
python main.py "создай статью с названием Овощи полезны для здоровья"
```

Other examples:

```bash
python main.py "get article with id 1"
python main.py "update article 1: set title to Healthy Vegetables Daily"
```

## Response contract

Every agent reply uses this fixed format:

```text
Status: success | error
Action: <описание действия>
Data: <результат API>
Errors: <если есть>
```

## Project layout

| Path | Role |
|------|------|
| `main.py` | CLI entrypoint |
| `agent.py` | LM Studio LLM + tool-calling loop |
| `tools_articles.py` | LangChain `StructuredTool` with an explicit HTTP wrapper |
| `articles_api.py` | HTTP client for create/get/update |
| `prompts.py` | System prompt (role, restrictions, tool rules) + contract helpers |
| `config.py` | Env-based settings |
| `verify.py` | Runs ≥5 sample queries (≥3 real API-tool calls) |

## Verification

```bash
python verify.py
```

This runs six sample user queries. If LM Studio is reachable, the full agent path is used; otherwise a deterministic fallback still exercises the same LangChain tool and performs real HTTP calls to JSONPlaceholder. Results are written to `verification_results.json` (gitignored).

## System prompt highlights

The agent is constrained to:

- Role: Articles API operator only
- Allowed: create, get, update via `articles_api`
- Forbidden: delete, unrelated Q&A, inventing API data, other tools/APIs
- Must call the tool before answering API requests and return the fixed response contract
