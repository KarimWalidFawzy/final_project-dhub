"""Small orchestration primitives used by the research agent."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM

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
        self.model = AutoModelForCausalLM.from_pretrained("gpt3")  # Placeholder for a model or LLM if needed
        self.tokenizer = AutoTokenizer.from_pretrained("gpt3")  # Placeholder for a tokenizer if needed

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