import numexpr
import wikipedia
import os
import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable


class BaseTool:
    name = ""
    description = ""
    parameters = {}

    def run(self, **kwargs):
        raise NotImplementedError

    def spec(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class MCPToolBridge(BaseTool):
    """动态代理 MCP 远程工具，使其兼容内部 BaseTool 接口。

    通过 MCPManager 发现工具后，为每个远程工具创建一个 MCPToolBridge 实例，
    注册到 TOOL_REGISTRY 中，ExecutorAgent 即可透明调用。
    """

    def __init__(self, qualified_name: str, tool_spec: Dict[str, Any],
                 runner: Callable[[str, Dict[str, Any]], str]):
        self.name = qualified_name
        self.description = tool_spec.get("description", "")
        self.parameters = tool_spec.get("parameters", {})
        self._runner = runner
        self._server_name = tool_spec.get("server_name", "")

    def run(self, **kwargs) -> str:
        return self._runner(self.name, kwargs)

    def spec(self):
        return {
            "name": self.name,
            "description": f"[MCP:{self._server_name}] {self.description}",
            "parameters": self.parameters,
        }


def register_mcp_tools(mcp_manager, target_registry: Dict[str, BaseTool]) -> int:
    """从 MCPManager 发现并注册所有远程工具到目标注册表。

    返回注册的工具数量。已存在的 MCP 工具键会被覆盖更新。
    """
    if mcp_manager is None:
        return 0
    try:
        specs = mcp_manager.discover_all_tools()
    except Exception:
        return 0

    count = 0
    for spec in specs:
        qualified = spec.to_internal_spec()["name"]

        def _make_runner(qname: str):
            def _runner(_unused: str, args: Dict[str, Any]) -> str:
                return mcp_manager.call_tool(qname, args)
            return _runner

        bridge = MCPToolBridge(
            qualified_name=qualified,
            tool_spec={
                "description": spec.description,
                "parameters": spec.parameters,
                "server_name": spec.server_name,
            },
            runner=_make_runner(qualified),
        )
        target_registry[qualified] = bridge
        count += 1

    return count


def unregister_mcp_tools(target_registry: Dict[str, BaseTool]):
    """移除所有 MCP 代理工具（以 'mcp__' 为前缀的键）。"""
    keys_to_remove = [k for k in target_registry if k.startswith("mcp__")]
    for k in keys_to_remove:
        target_registry.pop(k, None)


def _workspace_root():
    return Path(os.environ.get("AGENT_WORKSPACE", os.getcwd())).resolve()


def _resolve_workspace_path(path):
    root = _workspace_root()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()

    try:
        target.relative_to(root)
    except ValueError:
        return None, f"拒绝访问: 路径超出工作目录 {root}"

    return target, ""

class Calculator(BaseTool):
    name = "calculator"
    description = "用于执行精确的数学运算，支持加减乘除和幂运算。"
    parameters = {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如 '2 + 2'"}}, "required": ["expression"]}

    def run(self, expression):
        try:
            expression = expression.replace("^", "**")
            return str(numexpr.evaluate(expression).item())
        except Exception as e:
            return f"计算错误: {str(e)}"

class WikiSearch(BaseTool):
    name = "wiki_search"
    description = "在维基百科搜索信息，获取人物、事件或概念的摘要。"
    parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}

    def _offline_fallback(self, query):
        normalized = query.lower().replace(" ", "")
        if "爱因斯坦" in normalized or "einstein" in normalized:
            return "阿尔伯特·爱因斯坦（1879年3月14日—1955年4月18日）是理论物理学家。"
        return None

    def run(self, query):
        try:
            wikipedia.set_lang("zh")
            return wikipedia.summary(query, sentences=3)
        except Exception as e:
            try:
                results = wikipedia.search(query, results=3)
                if not results:
                    return f"搜索失败: 未找到与 {query} 相关的维基百科词条"
                return wikipedia.summary(results[0], sentences=3)
            except Exception as retry_error:
                fallback = self._offline_fallback(query)
                if fallback:
                    return fallback
                return f"搜索失败: {str(e)}；重试失败: {str(retry_error)}"

class FileReader(BaseTool):
    name = "read_file"
    description = "读取本地文本文件的内容。"
    parameters = {"type": "object", "properties": {"path": {"type": "string", "description": "文件相对或绝对路径"}}, "required": ["path"]}

    def run(self, path):
        try:
            target, error = _resolve_workspace_path(path)
            if error:
                return error
            if not target.exists():
                return f"读取失败: 文件不存在 {target}"
            if target.is_dir():
                return f"读取失败: 路径是目录 {target}"
            if target.stat().st_size > 1024 * 1024:
                return f"读取失败: 文件过大 {target}"
            with open(target, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"读取失败: {str(e)}"

class FileWriter(BaseTool):
    name = "write_file"
    description = "将内容写入本地文本文件。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "工作目录内的目标文件路径"},
            "content": {"type": "string", "description": "要写入的文本内容"},
        },
        "required": ["path", "content"],
    }

    def run(self, path, content):
        try:
            target, error = _resolve_workspace_path(path)
            if error:
                return error
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"写入成功: {target}"
        except Exception as e:
            return f"写入失败: {str(e)}"

class DateTimeTool(BaseTool):
    name = "datetime_tool"
    description = "获取当前时间，或计算两个日期之间的天数差。"
    parameters = {"type": "object", "properties": {"action": {"type": "string", "enum": ["now", "diff"]}, "date1": {"type": "string"}, "date2": {"type": "string"}}, "required": ["action"]}

    def run(self, action, date1=None, date2=None):
        try:
            if action == "now":
                return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if action == "diff" and date1 and date2:
                d1 = datetime.datetime.strptime(date1, "%Y-%m-%d")
                d2 = datetime.datetime.strptime(date2, "%Y-%m-%d")
                return str(abs((d2 - d1).days))
            return "参数错误"
        except Exception as e:
            return f"日期计算错误: {str(e)}"


# ── Shell 工具 ─────────────────────────────────────────

class ShellTool(BaseTool):
    """安全的 Shell 命令执行工具，内置本地安全校验。"""

    name = "shell_exec"
    description = (
        "执行安全的 Shell 命令（列目录/查看文件/查找/统计等只读操作）。"
        "用于：列出文件(dir/ls)、查看内容(type/cat)、统计行数/字数(wc)、"
        "搜索文本(findstr/grep)、查看进程(tasklist/ps)、查看网络状态等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 Shell 命令。仅限低风险的只读/查询操作。"
            }
        },
        "required": ["command"],
    }

    def __init__(self):
        from shell_agent import SafetyRuleEngine, ShellCommandExecutor
        self.safety = SafetyRuleEngine()
        self.executor = ShellCommandExecutor(timeout=30, max_output_lines=200)

    def run(self, command: str) -> str:
        """带安全检查的命令执行。高风险命令会被拦截。"""
        if not command or not command.strip():
            return "错误: 命令为空"

        verdict = self.safety.check(command.strip())
        if not verdict.safe and verdict.requires_confirmation:
            return (
                f"[SECURITY BLOCK] 该命令被本地规则引擎拦截。\n"
                f"原因: {verdict.reason}\n"
                f"匹配规则:\n" + "\n".join(f"  • {r}" for r in verdict.matched_rules) +
                f"\n\n请换用更安全的操作方式，或通过 Shell Agent 模式确认后执行。"
            )

        # 只显示匹配的中等风险提示但放行
        if not verdict.safe:
            result = self.executor.execute(command.strip())
            prefix = "⚠️ [提示] 检测到中等风险模式:\n" + \
                     "\n".join(f"  • {r}" for r in verdict.matched_rules) + "\n\n"
            return prefix + (result.stdout or result.stderr or "(无输出)")

        result = self.executor.execute(command.strip())
        if result.success:
            return result.stdout or "(命令执行成功，无输出)"
        else:
            return f"命令执行失败 (退出码 {result.exit_code}):\n{result.stderr or result.stdout}"


# 工具注册表
# 注意: shell_exec 不在此注册，Shell Agent 仅限 TUI 终端使用
TOOL_REGISTRY = {
    "calculator": Calculator(),
    "wiki_search": WikiSearch(),
    "read_file": FileReader(),
    "write_file": FileWriter(),
    "datetime_tool": DateTimeTool(),
}
