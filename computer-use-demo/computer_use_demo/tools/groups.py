from dataclasses import dataclass

from .base import BaseAnthropicTool
from .bash import BashTool20250124
from .computer import ComputerTool20250124
from .edit import EditTool20250429


@dataclass(frozen=True, kw_only=True)
class ToolGroup:
    version: str
    tools: list[type[BaseAnthropicTool]]
    beta_flag: str


COMPUTER_USE_20250429 = ToolGroup(
    version="computer_use_20250429",
    tools=[ComputerTool20250124, EditTool20250429, BashTool20250124],
    beta_flag="computer-use-2025-01-24",
)
