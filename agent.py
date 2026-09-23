"""Small orchestration primitives used by the research agent."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:  # pragma: no cover - optional dependency for offline generation
    AutoTokenizer = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]


@dataclass
class StepResult:
    name: str
    output: Any = None
    error: Optional[str] = None


class Agent:
    """Run named functions in order and retain their outputs."""

    def __init__(self, name: str = "research-agent") -> None:
        self.name = name
        self.steps: List[Callable[[Dict[str, Any]], Any]] = []
        self.history: List[StepResult] = []
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def load_model(self, model_name: Optional[str] = None) -> tuple[Any, Any]:
        """Load a local Hugging Face model only when it is actually needed."""
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is required to load a local model")
        resolved_name = model_name or os.getenv("HF_MODEL") or "distilgpt2"
        self.model = AutoModelForCausalLM.from_pretrained(resolved_name)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        return self.model, self.tokenizer

    def add_step(self, name: str, function: Callable[[Dict[str, Any]], Any]) -> "Agent":
        function.step_name = name  # type: ignore[attr-defined]
        self.steps.append(function)
        return self

    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context or {})
        self.history.clear()
        for function in self.steps:
            name = getattr(function, "step_name", function.__name__)
            try:
                result = function(state)
                if isinstance(result, dict):
                    state.update(result)
                else:
                    state[name] = result
                self.history.append(StepResult(name=name, output=result))
            except Exception as exc:
                self.history.append(StepResult(name=name, error=str(exc)))
                raise RuntimeError(f"Agent step '{name}' failed: {exc}") from exc
        return state


class MultiAgentSystem:
    """Coordinate specialized agents through a shared task state."""

    def __init__(self, name: str = "dhub-multi-agent-system") -> None:
        self.name = name
        self.agents: List[tuple[str, Callable[[Dict[str, Any]], Any]]] = []
        self.history: List[StepResult] = []

    def register(self, name: str, function: Callable[[Dict[str, Any]], Any]) -> "MultiAgentSystem":
        self.agents.append((name, function))
        return self

    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context or {})
        self.history.clear()
        for name, function in self.agents:
            try:
                result = function(state)
                if isinstance(result, dict):
                    state.update(result)
                else:
                    state[name] = result
                self.history.append(StepResult(name=name, output=result))
            except Exception as exc:
                self.history.append(StepResult(name=name, error=str(exc)))
                raise RuntimeError(f"Agent '{name}' failed: {exc}") from exc
        return state


agent = Agent