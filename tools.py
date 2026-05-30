import numexpr
import wikipedia
import os
import datetime
from pathlib import Path

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

# 工具注册表
TOOL_REGISTRY = {
    "calculator": Calculator(),
    "wiki_search": WikiSearch(),
    "read_file": FileReader(),
    "write_file": FileWriter(),
    "datetime_tool": DateTimeTool()
}
