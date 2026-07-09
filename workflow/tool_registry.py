from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional

from workflow.action_models import ActionItem, ActionType


ToolHandler = Callable[[ActionItem, Any, Any], Awaitable[Any]]


class ToolRegistry:
    def __init__(self):
        self._handlers: Dict[ActionType, ToolHandler] = {}

    def register(self, action_type: ActionType | str, handler: ToolHandler) -> None:
        self._handlers[ActionType(action_type)] = handler

    def resolve(self, action_type: ActionType | str) -> Optional[ToolHandler]:
        return self._handlers.get(ActionType(action_type))

    async def execute(self, action: ActionItem, runtime: Any, evidence_store: Any) -> Any:
        handler = self.resolve(action.type)
        if handler is None:
            raise KeyError(f"未注册动作处理器: {action.type.value}")
        return await handler(action, runtime, evidence_store)

    def list_actions(self) -> List[str]:
        return [action_type.value for action_type in self._handlers]
