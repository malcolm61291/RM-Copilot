"""Importing this package registers every tool."""
from crm_agent.tools import analytics_tools, crm_tools  # noqa: F401
from crm_agent.tools.registry import ToolError, call_tool, catalogue

__all__ = ["call_tool", "catalogue", "ToolError"]
