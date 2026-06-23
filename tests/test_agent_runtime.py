import unittest

from src.agent_runtime import run_agent_task


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, generation_config):
        return {"text": self.responses.pop(0)}


class AgentRuntimeTests(unittest.TestCase):
    def test_scores_tool_sequence_and_error_code(self):
        adapter = FakeAdapter(
            [
                '{"type":"tool_call","name":"get_order","arguments":{"order_id":"O-1001"}}',
                '{"type":"tool_call","name":"request_return","arguments":{"order_id":"O-1001","item_ids":["lamp"]}}',
                '{"type":"final","answer":"submitted"}',
            ]
        )
        task = {
            "task_id": "T",
            "language": "en",
            "category": "multi_step_workflow",
            "prompt": "Return lamp",
            "expected": {"tool_sequence": ["get_order", "request_return"]},
        }
        result = run_agent_task(adapter, task, {})
        self.assertTrue(result["task_success"])

    def test_scores_argument_subset_and_recovery_error(self):
        adapter = FakeAdapter(
            [
                '{"type":"tool_call","name":"calculate_refund","arguments":{"order_id":"O-1001","item_ids":["mouse"]}}',
                '{"type":"final","answer":"item is not in the order"}',
            ]
        )
        task = {
            "task_id": "T",
            "language": "en",
            "category": "recovery_after_tool_error",
            "prompt": "Recover after bad item",
            "expected": {
                "first_tool": "calculate_refund",
                "first_arguments": {"order_id": "O-1001", "item_ids": ["mouse"]},
                "error_code": "bad_items",
                "final_contains": "item",
            },
        }
        result = run_agent_task(adapter, task, {})
        self.assertTrue(result["task_success"])


if __name__ == "__main__":
    unittest.main()
