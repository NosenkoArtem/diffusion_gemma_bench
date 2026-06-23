import unittest

from src.minitoolagent import AgentProtocolError, parse_response


class AgentParserTests(unittest.TestCase):
    def test_accepts_tool_call(self):
        parsed = parse_response('{"type":"tool_call","name":"get_order","arguments":{"order_id":"O-1001"}}')
        self.assertEqual(parsed["name"], "get_order")

    def test_accepts_final(self):
        parsed = parse_response('{"type":"final","answer":"done"}')
        self.assertEqual(parsed["answer"], "done")

    def test_rejects_markdown_wrapped_json(self):
        with self.assertRaises(AgentProtocolError):
            parse_response('```json\n{"type":"final","answer":"done"}\n```')

    def test_rejects_unknown_tool(self):
        with self.assertRaises(AgentProtocolError):
            parse_response('{"type":"tool_call","name":"shell","arguments":{"cmd":"ls"}}')


if __name__ == "__main__":
    unittest.main()
