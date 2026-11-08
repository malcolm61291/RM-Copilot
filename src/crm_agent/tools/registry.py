"""Tool registry. A tool = name + description + typed input schema + function.

- Inputs are validated with Pydantic before execution (the LLM never calls raw functions).
- Every call is traced (args, latency, result summary).
- `catalogue()` exports JSON schemas, used by the LLM planner and ready for native
  function-calling / MCP exposure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from crm_agent.observability.tracer import span


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Any]
    side_effect: bool = False  # True = changes the outside world; needs human approval upstream


_REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, args_model: type[BaseModel], side_effect: bool = False):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate tool name: {name}")
        _REGISTRY[name] = Tool(name, description, args_model, fn, side_effect)
        return fn
    return decorator


def call_tool(name: str, **kwargs: Any) -> Any:
    if name not in _REGISTRY:
        raise ToolError(f"Unknown tool '{name}'")
    t = _REGISTRY[name]
    try:
        args = t.args_model(**kwargs)
    except ValidationError as exc:
        raise ToolError(f"Invalid arguments for {name}: {exc}") from exc
    with span("tool", name, **_brief_args(args.model_dump())) as s:
        result = t.fn(**args.model_dump())
        s["result"] = _brief_result(result)
        return result


def catalogue() -> list[dict]:
    return [{"name": t.name, "description": t.description, "side_effect": t.side_effect,
             "input_schema": t.args_model.model_json_schema()} for t in _REGISTRY.values()]


def _brief_args(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if isinstance(v, list) and len(v) > 5:
            out[k] = f"[{len(v)} items]"
        elif isinstance(v, dict) and len(v) > 8:
            out[k] = f"{{{len(v)} keys}}"
        else:
            out[k] = v
    return out


def _brief_result(result: Any) -> Any:
    if isinstance(result, list):
        return f"{len(result)} items"
    if isinstance(result, dict) and len(result) > 8:
        return f"{len(result)} entries"
    return result
