"""LLM providers for D-Hub with a deterministic offline fallback."""

import os
from functools import lru_cache
from importlib import import_module
from typing import Optional
from summarizer import summarize


@lru_cache(maxsize=1)
def _get_groq_client():
    """Create the Groq client only when a Groq request is actually needed."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        groq_module = import_module("groq")
    except ImportError:
        return None
    return groq_module.Groq(api_key=api_key)


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Send a message to Groq and return the text response.

    Args:
        system_prompt: Defines the agent's role and behavior.
        user_message:  The input content for the agent to process.

    Returns:
        The agent's text response.
    """
    client = _get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY and the groq package are required")
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()

def _prompt(text: str, task: str) -> str:
    return (
        "Answer the research task using only the evidence below. "
        "State uncertainty when evidence is insufficient.\n"
        f"Task: {task}\nEvidence: {text}"
    )


@lru_cache(maxsize=1)
def _get_transformer_generator():
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers is required for local LLM summaries") from exc

    model_name = os.getenv("HF_MODEL", "google/flan-t5-small")
    return pipeline("text2text-generation", model=model_name)


def _openai_summary(text: str, task: str) -> Optional[str]:
    """Call the OpenAI Python API when an API key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        return None


def _groq_summary(text: str, task: str) -> Optional[str]:
    """Summarize evidence through the Groq chat-completions API."""
    client = _get_groq_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {
                    "role": "system",
                    "content": "Answer the research task using only the supplied evidence. State uncertainty when evidence is insufficient.",
                },
                {"role": "user", "content": _prompt(text, task)},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except (ImportError, OSError, ValueError, IndexError):
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
    """Run a cached local Hugging Face text-to-text model when explicitly requested."""
    try:
        generator = _get_transformer_generator()
        result = generator(_prompt(text, task), max_new_tokens=180, do_sample=False)
        generated = result[0].get("generated_text", "")
        return generated.strip() or None
    except (ImportError, OSError, RuntimeError, ValueError, IndexError):
        return None


def summarize_with_llm(text: str, task: str, max_sentences: int = 5) -> str:
    """Summarize using Groq, OpenAI, Transformers, or extractive fallback.

    Set ``LLM_PROVIDER`` to ``groq``, ``openai``, ``transformers``, or
    ``extractive``. The default ``auto`` uses Groq when configured, then
    OpenAI, local Transformers, and finally the offline extractive fallback.
    """
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider not in {"auto", "groq", "openai", "transformers", "extractive"}:
        raise ValueError("LLM_PROVIDER must be auto, groq, openai, transformers, or extractive")
    if provider in {"auto", "groq"}:
        result = _groq_summary(text, task)
        if result or provider == "groq":
            return result or summarize(text, task, max_sentences)
    if provider in {"auto", "openai"}:
        result = _openai_summary(text, task)
        if result or provider == "openai":
            return result or summarize(text, task, max_sentences)
    if provider == "transformers" or (provider == "auto" and os.getenv("HF_MODEL")):
        result = _transformers_summary(text, task)
        if result:
            return result
    return summarize(text, task, max_sentences)
