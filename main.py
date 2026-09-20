#!/usr/bin/env python3
"""CLI entrypoint for the Articles API LangChain agent."""

from __future__ import annotations

import argparse
import sys

from agent import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Articles API Operator — LangChain agent via local LM Studio",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help='User request, e.g. \'создай статью с названием Овощи полезны для здоровья\'',
    )
    args = parser.parse_args(argv)

    if not args.query:
        parser.error("query is required, e.g. python main.py \"создай статью с названием Овощи\"")

    print(run_agent(args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
