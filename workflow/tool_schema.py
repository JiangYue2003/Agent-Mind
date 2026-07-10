"""Tool contracts used by workflow routing before an action is created."""
from typing import Dict, List


TOOL_REQUIRED_SLOTS: Dict[str, List[str]] = {
    "knowledge_search": [],
    "order_lookup": ["order_id"],
    "shipment_track": ["order_id"],
    "refund_create": ["order_id"],
}


def required_slots_for_tools(tools: List[str]) -> List[str]:
    slots: List[str] = []
    for tool in tools:
        for slot in TOOL_REQUIRED_SLOTS.get(tool, []):
            if slot not in slots:
                slots.append(slot)
    return slots
