import unittest

from src.minitoolagent import AgentProtocolError, call_tool, validate_tool_arguments


class ToolTests(unittest.TestCase):
    def test_get_order_is_deterministic(self):
        first = call_tool("get_order", {"order_id": "O-1001"})
        second = call_tool("get_order", {"order_id": "O-1001"})
        self.assertEqual(first, second)
        self.assertTrue(first["ok"])

    def test_schema_rejects_extra_keys(self):
        with self.assertRaises(AgentProtocolError):
            validate_tool_arguments("get_order", {"order_id": "O-1001", "extra": True})

    def test_restricted_account_returns_structured_error(self):
        result = call_tool(
            "update_notification_setting",
            {"user_id": "U-2", "setting": "sms", "value": False},
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "policy_restricted")


if __name__ == "__main__":
    unittest.main()
