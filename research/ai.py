"""Optional, read-only AI interpretation; never executes model-generated code."""
from __future__ import annotations

import json
from openai import OpenAI


def interpret(summary: dict, question: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    if not api_key:
        raise ValueError("Provide an OpenAI API key to enable AI interpretation.")
    client = OpenAI(api_key=api_key, timeout=30.0, max_retries=1)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "You are a careful data science research assistant. Use only the provided "
                "aggregate statistics. Never invent data, causal conclusions, significance "
                "tests, sources or performance results. Explicitly discuss limitations. "
                "Do not provide executable code unless specifically requested.")},
            {"role": "user", "content": "Dataset summary (aggregate statistics only):\n"
             + json.dumps(summary, default=str)[:12000]
             + "\nResearch question: " + question[:2000]},
        ],
    )
    return response.choices[0].message.content or "No interpretation returned."
