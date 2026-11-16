"""Backward-compatible research assistant interface.

The application intentionally does not execute arbitrary LLM-generated Python.
Use research.ai.interpret for read-only aggregate interpretation.
"""
from research.ai import interpret


def create_ds_agent(tools=None):
    """Legacy entry point removed to avoid unsupported LangChain imports.

    Use interpret(summary, question, api_key) for safe research commentary.
    """
    raise NotImplementedError(
        "The unsafe legacy tool-executing agent has been replaced. "
        "Use research.ai.interpret for aggregate, read-only analysis."
    )
