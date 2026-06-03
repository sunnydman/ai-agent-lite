import os
import sys
import json
import types
import shutil
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_RUNTIME = PROJECT_ROOT / "tests_runtime"


def fresh_runtime_dir(name: str) -> Path:
    path = TEST_RUNTIME / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_dependency_stubs():
    """Keep core tests independent from optional runtime packages."""
    openai = types.ModuleType("openai")
    openai.OpenAI = object
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

from agents import ExecutorAgent, PlannerAgent, OrchestratorAgent, AgentMessage
from memory import MemoryManager
import tools
from tools import FileWriter, WikiSearch


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, stream=False):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("FakeLLM has no reply left")
        reply = self.replies.pop(0)
        if callable(reply):
            return reply(messages)
        return reply


class AgentCoreTests(unittest.TestCase):
    def test_tool_description_includes_parameter_schema(self):
        executor = ExecutorAgent(FakeLLM([]), "rules")

        tools = json.loads(executor.tools_desc)
        calculator = next(t for t in tools if t["name"] == "calculator")

        self.assertIn("parameters", calculator)
        self.assertEqual(calculator["parameters"]["required"], ["expression"])

    def test_file_writer_rejects_path_outside_workspace(self):
        parent = fresh_runtime_dir("file_writer")
        workspace = parent / "workspace"
        workspace.mkdir()
        old_cwd = os.getcwd()
        try:
            os.chdir(workspace)
            result = FileWriter().run("../outside.txt", "secret")
        finally:
            os.chdir(old_cwd)

        self.assertIn("拒绝", result)
        self.assertFalse((parent / "outside.txt").exists())

    def test_wiki_search_has_einstein_fallback_when_backend_fails(self):
        old_summary = tools.wikipedia.summary
        old_search = tools.wikipedia.search
        try:
            tools.wikipedia.summary = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("backend failed"))
            tools.wikipedia.search = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("backend failed"))

            result = WikiSearch().run("爱因斯坦")
        finally:
            tools.wikipedia.summary = old_summary
            tools.wikipedia.search = old_search

        self.assertIn("1879年3月14日", result)
        self.assertIn("1955年4月18日", result)

    def test_orchestrator_injects_retrieved_memory_context(self):
        memory = MemoryManager(":memory:")
        memory.save("s1", "assistant", "project codename alpha")

        llm = FakeLLM([
            "NO",
            "The project codename is alpha.",
        ])
        orchestrator = OrchestratorAgent(
            llm,
            PlannerAgent(llm, ExecutorAgent(llm, "rules")),
            memory,
        )
        msg = AgentMessage("User", "Orchestrator", "codename", {"session_id": "s1"})

        orchestrator.process(msg)

        self.assertIn("memory_context", msg.metadata)
        self.assertIn("project codename alpha", msg.metadata["memory_context"])

    def test_orchestrator_remember_command_saves_memory_without_llm(self):
        memory = MemoryManager(":memory:")
        llm = FakeLLM([])
        orchestrator = OrchestratorAgent(
            llm,
            PlannerAgent(llm, ExecutorAgent(llm, "rules")),
            memory,
        )

        response = orchestrator.process(
            AgentMessage(
                "User",
                "Orchestrator",
                "\u8bb0\u4f4f\uff1a\u6211\u7684\u9879\u76ee\u4ee3\u53f7\u662f alpha-a",
                {"session_id": "s-memory"},
            )
        )

        self.assertIn("\u6211\u8bb0\u4f4f\u4e86", response.content)
        self.assertIn("alpha-a", response.content)
        self.assertEqual(len(llm.calls), 0)
        saved = "\n".join(item["content"] for item in memory.get_recent("s-memory", 5))
        self.assertIn("alpha-a", saved)

    def test_orchestrator_recalls_project_codename_without_llm(self):
        memory = MemoryManager(":memory:")
        memory.save("s-memory", "memory", "\u6211\u7684\u9879\u76ee\u4ee3\u53f7\u662f alpha-a")
        llm = FakeLLM([])
        orchestrator = OrchestratorAgent(
            llm,
            PlannerAgent(llm, ExecutorAgent(llm, "rules")),
            memory,
        )

        response = orchestrator.process(
            AgentMessage(
                "User",
                "Orchestrator",
                "\u6211\u7684\u9879\u76ee\u4ee3\u53f7\u662f\u4ec0\u4e48\uff1f",
                {"session_id": "s-memory"},
            )
        )

        self.assertIn("alpha-a", response.content)
        self.assertEqual(len(llm.calls), 0)
        self.assertNotIn("\u65e0\u6cd5\u6267\u884c", response.content)

    def test_executor_accepts_json_action_format(self):
        llm = FakeLLM([
            json.dumps({
                "thought": "Need exact math.",
                "action": "calculator",
                "action_input": {"expression": "2 + 2"},
            }),
            json.dumps({
                "thought": "Tool returned the answer.",
                "final_answer": "4",
            }),
        ])
        executor = ExecutorAgent(llm, "rules")

        response = executor.process(AgentMessage("User", "Executor", "calculate 2 + 2"))

        self.assertEqual(response.content, "4")

    def test_executor_stops_repeated_tool_call_and_returns_observation(self):
        repeated_action = json.dumps({
            "thought": "Need exact math.",
            "action": "calculator",
            "action_input": {"expression": "2 + 2"},
        })
        llm = FakeLLM([repeated_action, repeated_action])
        executor = ExecutorAgent(llm, "rules")

        response = executor.process(AgentMessage("User", "Executor", "calculate 2 + 2"))

        self.assertEqual(response.content, "计算结果：4")
        self.assertIn("重复工具调用", "\n".join(response.metadata["trace"]))

    def test_executor_handles_explicit_write_then_read_without_llm_loop(self):
        workspace = fresh_runtime_dir("write_then_read")
        old_cwd = os.getcwd()
        try:
            os.chdir(workspace)
            llm = FakeLLM([])
            executor = ExecutorAgent(llm, "rules")

            response = executor.process(
                AgentMessage(
                    "User",
                    "Executor",
                    "\u628a\u201chello agent\u201d\u5199\u5165 notes/demo.txt\uff0c\u7136\u540e\u8bfb\u53d6\u8fd9\u4e2a\u6587\u4ef6",
                )
            )
        finally:
            os.chdir(old_cwd)

        self.assertEqual((workspace / "notes" / "demo.txt").read_text(encoding="utf-8"), "hello agent")
        self.assertIn("hello agent", response.content)
        self.assertEqual(len(llm.calls), 0)
        self.assertNotIn("达到最大循环次数", "\n".join(response.metadata["trace"]))

    def test_executor_answers_lifespan_from_wiki_observation_without_looping(self):
        llm = FakeLLM([
            json.dumps({
                "thought": "Need biographical dates.",
                "action": "wiki_search",
                "action_input": {"query": "爱因斯坦"},
            }),
        ])
        executor = ExecutorAgent(llm, "rules")

        response = executor.process(AgentMessage("User", "Executor", "爱因斯坦活了多少岁"))

        self.assertIn("76", response.content)
        self.assertIn("岁", response.content)
        self.assertEqual(len(llm.calls), 0)
        self.assertNotIn("达到最大循环次数", "\n".join(response.metadata["trace"]))

    def test_executor_answers_lifespan_without_wrong_tool_misfire(self):
        llm = FakeLLM([
            json.dumps({
                "thought": "I picked the wrong tool.",
                "action": "write_file",
                "action_input": {},
            }),
        ])
        executor = ExecutorAgent(llm, "rules")

        response = executor.process(AgentMessage("User", "Executor", "爱因斯坦活了多少岁"))

        self.assertIn("76", response.content)
        self.assertIn("岁", response.content)
        self.assertEqual(len(llm.calls), 0)
        self.assertNotIn("FileWriter.run", response.content)
        self.assertNotIn("达到最大循环次数", "\n".join(response.metadata["trace"]))


if __name__ == "__main__":
    unittest.main()
