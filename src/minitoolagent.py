"""Deterministic MiniToolAgent v1 primitives.

The benchmark uses local tools only. This module contains strict parsing,
schema validation, and deterministic tool execution so model quality is judged
without network calls, shell access, filesystem access, retries, or JSON repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


class AgentProtocolError(ValueError):
    """Raised when a model response violates the strict JSON/tool protocol."""


ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """Small JSON-schema-like contract for a deterministic local tool."""

    name: str
    required: dict[str, type | tuple[type, ...]]
    handler: ToolFn


STATE = {
    "orders": {
        "O-1001": {"user_id": "U-1", "status": "delivered", "items": ["book", "lamp"], "returnable": True},
        "O-1002": {"user_id": "U-1", "status": "processing", "items": ["keyboard"], "returnable": False},
        "O-2001": {"user_id": "U-2", "status": "delivered", "items": ["mug"], "returnable": True},
    },
    "accounts": {
        "U-1": {"region": "US", "notifications": {"email": True, "sms": False}, "restricted": False},
        "U-2": {"region": "RU", "notifications": {"email": True, "sms": True}, "restricted": True},
    },
    "flights": {
        ("SFO", "JFK", "2026-07-01"): [{"flight_id": "F-77", "seats": 3}],
        ("MOW", "IST", "2026-07-02"): [{"flight_id": "F-88", "seats": 0}],
    },
    "bookings": {"B-9": {"user_id": "U-1", "flight_id": "F-70", "changeable": True}},
}


def _ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "data": data}


def _err(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def get_order(args: dict[str, Any]) -> dict[str, Any]:
    order = STATE["orders"].get(args["order_id"])
    return _ok(order | {"order_id": args["order_id"]}) if order else _err("not_found", "order not found")


def list_orders(args: dict[str, Any]) -> dict[str, Any]:
    orders = [
        {"order_id": order_id, **order}
        for order_id, order in STATE["orders"].items()
        if order["user_id"] == args["user_id"]
    ]
    return _ok({"orders": orders})


def calculate_refund(args: dict[str, Any]) -> dict[str, Any]:
    order = STATE["orders"].get(args["order_id"])
    if not order:
        return _err("not_found", "order not found")
    if not order["returnable"]:
        return _err("not_returnable", "order is not returnable")
    missing = [item for item in args["item_ids"] if item not in order["items"]]
    if missing:
        return _err("bad_items", "one or more items are not in the order")
    return _ok({"amount": 10 * len(args["item_ids"]), "currency": "USD"})


def request_return(args: dict[str, Any]) -> dict[str, Any]:
    refund = calculate_refund(args)
    if not refund["ok"]:
        return refund
    return _ok({"return_id": f"R-{args['order_id']}", "status": "submitted"})


def search_flights(args: dict[str, Any]) -> dict[str, Any]:
    key = (args["origin"], args["destination"], args["date"])
    return _ok({"flights": STATE["flights"].get(key, [])})


def change_booking(args: dict[str, Any]) -> dict[str, Any]:
    booking = STATE["bookings"].get(args["booking_id"])
    if not booking:
        return _err("not_found", "booking not found")
    if not booking["changeable"]:
        return _err("not_changeable", "booking cannot be changed")
    return _ok({"booking_id": args["booking_id"], "flight_id": args["new_flight_id"], "status": "changed"})


def get_account(args: dict[str, Any]) -> dict[str, Any]:
    account = STATE["accounts"].get(args["user_id"])
    return _ok(account | {"user_id": args["user_id"]}) if account else _err("not_found", "account not found")


def update_notification_setting(args: dict[str, Any]) -> dict[str, Any]:
    account = STATE["accounts"].get(args["user_id"])
    if not account:
        return _err("not_found", "account not found")
    if account["restricted"]:
        return _err("policy_restricted", "restricted accounts cannot be changed")
    setting = args["setting"]
    if setting not in account["notifications"]:
        return _err("bad_setting", "unknown notification setting")
    return _ok({"user_id": args["user_id"], "setting": setting, "value": args["value"]})


TOOLS: dict[str, ToolSpec] = {
    "get_order": ToolSpec("get_order", {"order_id": str}, get_order),
    "list_orders": ToolSpec("list_orders", {"user_id": str}, list_orders),
    "calculate_refund": ToolSpec("calculate_refund", {"order_id": str, "item_ids": list}, calculate_refund),
    "request_return": ToolSpec("request_return", {"order_id": str, "item_ids": list}, request_return),
    "search_flights": ToolSpec("search_flights", {"origin": str, "destination": str, "date": str}, search_flights),
    "change_booking": ToolSpec("change_booking", {"booking_id": str, "new_flight_id": str}, change_booking),
    "get_account": ToolSpec("get_account", {"user_id": str}, get_account),
    "update_notification_setting": ToolSpec(
        "update_notification_setting", {"user_id": str, "setting": str, "value": (bool, str)}, update_notification_setting
    ),
}


SYSTEM_PROMPT = (
    "You are a tool-using assistant. "
    "Return exactly one JSON object and no other text. "
    "Do not provide explanations, reasoning, markdown, or code fences. "
    "Use one tool call per turn when a tool is required."
)


def parse_response(raw_text: str) -> dict[str, Any]:
    """Parse one strict model response.

    The function deliberately rejects markdown, prefixed text, multiple JSON
    objects, and trailing commentary because the benchmark forbids repair.
    """

    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AgentProtocolError("invalid_json") from exc
    if not isinstance(obj, dict):
        raise AgentProtocolError("response_must_be_object")
    kind = obj.get("type")
    if kind == "final":
        if set(obj) != {"type", "answer"} or not isinstance(obj["answer"], str):
            raise AgentProtocolError("invalid_final_shape")
        return obj
    if kind == "tool_call":
        if set(obj) != {"type", "name", "arguments"}:
            raise AgentProtocolError("invalid_tool_call_shape")
        if not isinstance(obj["name"], str) or not isinstance(obj["arguments"], dict):
            raise AgentProtocolError("invalid_tool_call_fields")
        validate_tool_arguments(obj["name"], obj["arguments"])
        return obj
    raise AgentProtocolError("unknown_response_type")


def validate_tool_arguments(tool_name: str, args: dict[str, Any]) -> None:
    """Validate tool name and argument types against the local schema."""

    spec = TOOLS.get(tool_name)
    if not spec:
        raise AgentProtocolError("unknown_tool")
    missing = [name for name in spec.required if name not in args]
    extra = [name for name in args if name not in spec.required]
    if missing or extra:
        raise AgentProtocolError("schema_keys_mismatch")
    for name, expected_type in spec.required.items():
        if not isinstance(args[name], expected_type):
            raise AgentProtocolError(f"schema_type_mismatch:{name}")


def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a deterministic local tool after validation."""

    validate_tool_arguments(tool_name, args)
    return TOOLS[tool_name].handler(args)
