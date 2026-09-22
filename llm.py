"""LLM providers for D-Hub with a deterministic offline fallback."""

import os
from typing import Optional

from summarizer import summarize


def _prompt(text: str, task: str) -> str:
    return (
        "Answer the research task using only the evidence below. "
        "State uncertainty when evidence is insufficient.\n"
        f"Task: {task}\nEvidence: {text}"
    )


def _openai_summary(text: str, task: str) -> Optional[str]:
    """Call the OpenAI Python API when an API key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": _prompt(text, task)}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except (ImportError, OSError, ValueError, IndexError):
        return None


def _transformers_summary(text: str, task: str) -> Optional[str]:
    """Run a local Hugging Face text-to-text model when explicitly requested."""
    try:
        from transformers import pipeline

        generator = pipeline(
            "text2text-generation",
            model=os.getenv("HF_MODEL", "google/flan-t5-small"),
        )
        result = generator(_prompt(text, task), max_new_tokens=180, do_sample=False)
        generated = result[0].get("generated_text", "")
        return generated.strip() or None
    except (ImportError, OSError, RuntimeError, ValueError, IndexError):
        return None


def summarize_with_llm(text: str, task: str, max_sentences: int = 5) -> str:
    """Summarize using OpenAI, Transformers, or extractive fallback.

    Set ``LLM_PROVIDER`` to ``openai``, ``transformers``, or ``extractive``.
    The default ``auto`` uses OpenAI when configured, then local Transformers
    only when ``HF_MODEL`` is explicitly set, and otherwise stays offline.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider not in {"auto", "openai", "transformers", "extractive"}:
        raise ValueError("LLM_PROVIDER must be auto, openai, transformers, or extractive")
    if provider in {"auto", "openai"}:
        result = _openai_summary(text, task)
        if result or provider == "openai":
            return result or summarize(text, task, max_sentences)
    if provider == "transformers" or (provider == "auto" and os.getenv("HF_MODEL")):
        result = _transformers_summary(text, task)
        if result:
            return result
    return summarize(text, task, max_sentences)
