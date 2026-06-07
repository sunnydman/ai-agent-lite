import json
import platform
import sys
import time
import types
import unittest
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def install_dependency_stubs():
    openai = types.ModuleType("openai")

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    openai.OpenAI = FakeOpenAI
    openai.APIError = Exception
    openai.RateLimitError = Exception
    openai.APITimeoutError = Exception
    sys.modules.setdefault("openai", openai)

    class NumResult:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    numexpr = types.ModuleType("numexpr")
    numexpr.__version__ = "2.10.2"
    numexpr.evaluate = lambda expr: NumResult(eval(expr, {"__builtins__": {}}, {}))
    sys.modules.setdefault("numexpr", numexpr)

    wikipedia = types.ModuleType("wikipedia")
    wikipedia.set_lang = lambda lang: None
    wikipedia.summary = lambda query, sentences=3: "Albert Einstein was born in 1879 and died in 1955."
    wikipedia.search = lambda query, results=5: ["Albert Einstein"]
    sys.modules.setdefault("wikipedia", wikipedia)


install_dependency_stubs()

from agents import ExecutorAgent
from mcp_client import MCPManager, MCPServerConfig
from shell_agent import SafetyRuleEngine, ShellAgent, ShellCommandExecutor
from web_ui import parse_mcp_args


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, stream=False):
        if not self.replies:
            raise AssertionError("FakeLLM has no reply left")
        return self.replies.pop(0)


class PR2ExtensionTests(unittest.TestCase):
    def test_shell_safety_allows_only_read_only_commands(self):
        engine = SafetyRuleEngine()

        allowed = ["dir", "type AGENTS.md", "findstr Agent README.md"]
        blocked = [
            "mkdir demo",
            "echo hello > notes.txt",
            "dir & whoami",
            "curl https://example.com",
            "python -c \"print(1)\"",
            "type ..\\secret.txt",
            "Remove-Item -Recurse -Force C:\\Windows",
        ]

        for command in allowed:
            with self.subTest(command=command):
                self.assertTrue(engine.check(command).safe)

        for command in blocked:
            with self.subTest(command=command):
                verdict = engine.check(command)
                self.assertFalse(verdict.safe)
                self.assertFalse(verdict.requires_confirmation)

    def test_shell_agent_refuses_non_read_only_without_pending_confirmation(self):
        llm = FakeLLM([
            json.dumps({
                "intent": "run_command",
                "command": "mkdir demo",
                "reason": "create directory",
                "risk_level": "medium",
            })
        ])
        agent = ShellAgent(llm, cwd=str(PROJECT_ROOT))

        cmd_info, verdict, needs_confirm, result = agent.run_pipeline("创建 demo 目录")

        self.assertEqual(cmd_info["intent"], "run_command")
        self.assertFalse(verdict.safe)
        self.assertFalse(needs_confirm)
        self.assertIsNone(result)
        self.assertIsNone(agent.get_pending_command())

    def test_shell_agent_marks_auto_executed_read_only_command_as_unconfirmed(self):
        command = "dir" if platform.system() == "Windows" else "ls"
        llm = FakeLLM([
            json.dumps({
                "intent": "run_command",
                "command": command,
                "reason": "list files",
                "risk_level": "low",
            })
        ])
        agent = ShellAgent(llm, cwd=str(PROJECT_ROOT))

        _, verdict, needs_confirm, result = agent.run_pipeline("列出当前目录")

        self.assertTrue(verdict.safe)
        self.assertFalse(needs_confirm)
        self.assertIsNotNone(result)
        self.assertFalse(result.was_confirmed)

    def test_shell_streaming_command_respects_timeout(self):
        command = "python -c \"import time; time.sleep(2)\"" if platform.system() == "Windows" else "sleep 2"
        executor = ShellCommandExecutor(timeout=0.5, cwd=str(PROJECT_ROOT))

        start = time.monotonic()
        output = list(executor.execute_stream(command))
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 1.8)
        self.assertTrue(any("超时" in line or "timeout" in line.lower() for line in output))

    def test_mcp_connect_all_reuses_existing_live_client(self):
        manager = MCPManager([
            MCPServerConfig(name="demo", command="python", args=["demo_mcp_server.py"])
        ])
        first_transport = None
        try:
            self.assertEqual(manager.connect_all(), {"demo": True})
            first_client = manager._clients["demo"]
            first_transport = first_client.transport
            first_pid = first_transport._process.pid

            self.assertEqual(manager.connect_all(), {"demo": True})
            second_pid = manager._clients["demo"].transport._process.pid

            self.assertEqual(first_pid, second_pid)
        finally:
            if first_transport is not None and first_transport.is_alive:
                first_transport.stop()
            manager.disconnect_all()

    def test_executor_keeps_mcp_tools_instance_scoped(self):
        manager = MCPManager([
            MCPServerConfig(name="demo", command="python", args=["demo_mcp_server.py"])
        ])
        try:
            manager.connect_all()
            executor_with_mcp = ExecutorAgent(FakeLLM([]), "rules", mcp_manager=manager)
            ExecutorAgent(FakeLLM([]), "rules", mcp_manager=MCPManager())

            self.assertIn("mcp__demo__current_time", executor_with_mcp.tool_registry)
        finally:
            manager.disconnect_all()

    def test_demo_mcp_count_files_rejects_paths_outside_workspace(self):
        manager = MCPManager([
            MCPServerConfig(name="demo", command="python", args=["demo_mcp_server.py"])
        ])
        try:
            manager.connect_all()
            manager.discover_all_tools()
            result = manager.call_tool("mcp__demo__count_files", {"directory": "C:\\Windows"})
            self.assertIn("拒绝", result)
        finally:
            manager.disconnect_all()

    def test_tui_imports_and_web_ui_mcp_args_parser_preserves_quoted_paths(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            __import__("tui")

        self.assertEqual(
            parse_mcp_args('demo_mcp_server.py "--path with spaces"'),
            ["demo_mcp_server.py", "--path with spaces"],
        )


if __name__ == "__main__":
    unittest.main()
