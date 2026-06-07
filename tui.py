"""
TUI — Terminal User Interface for AI Agent Lite
基于 rich 库的终端交互界面，支持:
  - 流式对话（逐字符打字效果）
  - Trace 推理轨迹实时展示
  - 多会话管理
  - MCP 服务器配置与状态监控
  - 命令快捷操作 (/help, /clear, /mcp, /quit)

使用方法:
    python tui.py                        # 启动交互式终端
    python tui.py --api-key sk-xxx       # 直接指定 API Key
    python tui.py --mcp-config mcp.json  # 加载 MCP 配置文件
"""

import os
import sys
import time
import json
import uuid
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.align import Align
from rich.style import Style
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich.columns import Columns

from llm_client import LLMClient
from agents import OrchestratorAgent, PlannerAgent, ExecutorAgent, AgentMessage, AgentEvent
from memory import MemoryManager
from mcp_client import MCPManager, MCPServerConfig
from shell_agent import ShellAgent, SafetyRuleEngine, ShellCommandExecutor

# ── 常量 ───────────────────────────────────────────────

try:
    with open(PROJECT_ROOT / "AGENTS.md", "r", encoding="utf-8") as f:
        AGENTS_MD = f.read()
except FileNotFoundError:
    AGENTS_MD = "未找到 AGENTS.md 文件。"

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mcp_servers.json"

# ── 终端样式 ───────────────────────────────────────────

STYLE_TITLE = Style(color="#c15f3f", bold=True)
STYLE_SUBTITLE = Style(color="#756458", italic=True)
STYLE_TRACE = Style(color="#8f7d6f")
STYLE_TOOL_CALL = Style(color="#4a90d9", bold=True)
STYLE_TOOL_RESULT = Style(color="#6b5a4e")
STYLE_THOUGHT = Style(color="#5a7a4a", italic=True)
STYLE_ASSISTANT = Style(color="#2f241c")
STYLE_USER = Style(color="#c15f3f", bold=True)
STYLE_ERROR = Style(color="#cc3333", bold=True)
STYLE_MCP = Style(color="#9b59b6", bold=True)
STYLE_INFO = Style(color="#756458")

console = Console()


# ── 渲染工具函数 ───────────────────────────────────────

def _format_trace_line(line: str) -> Text:
    """根据 Trace 内容上色。"""
    text = Text(line)
    if "❌" in line or "异常" in line or "失败" in line:
        text.stylize(STYLE_ERROR)
    elif "🔧" in line:
        text.stylize(STYLE_TOOL_CALL)
    elif "👁️" in line:
        text.stylize(STYLE_TOOL_RESULT)
    elif "💭" in line:
        text.stylize(STYLE_THOUGHT)
    elif "✅" in line or "🎯" in line:
        text.stylize(STYLE_ASSISTANT)
    elif "🛠️" in line or "🔄" in line or "📥" in line or "🚀" in line:
        text.stylize(STYLE_TRACE)
    elif "💾" in line or "📚" in line or "🔎" in line:
        text.stylize(STYLE_TRACE)
    elif "🔴" in line or "🟢" in line:
        text.stylize(STYLE_MCP)
    else:
        text.stylize(STYLE_TRACE)
    return text


def render_banner():
    """渲染启动横幅。"""
    banner = Panel(
        Align.center(
            Text("AI Agent Lite\n", style=STYLE_TITLE, justify="center") +
            Text("轻量多 Agent 工作台 · TUI 终端版\n", style=STYLE_SUBTITLE, justify="center") +
            Text("支持 MCP 远程工具 · 规划 · 执行 · 长期记忆\n", style=STYLE_INFO, justify="center") +
            Text("\n命令: /help 查看帮助 | /clear 清空记忆 | /mcp 查看服务器 | /quit 退出",
                 style=Style(color="#9a887c")),
        ),
        border_style=Style(color="#c15f3f"),
        padding=(1, 2),
    )
    console.print(banner)


def render_mcp_status_table(mcp_manager: MCPManager) -> Table:
    """渲染 MCP 服务器状态表格。"""
    table = Table(title="MCP 服务器状态", style=STYLE_MCP, border_style=Style(color="#9b59b6"))
    table.add_column("名称", style="bold")
    table.add_column("状态")
    table.add_column("工具数")
    table.add_column("描述")

    status_map = mcp_manager.get_server_status()
    for name, s in status_map.items():
        if s["connected"]:
            state = Text("🟢 已连接", style=Style(color="#27ae60"))
        elif s["enabled"]:
            state = Text("🔴 未连接", style=Style(color="#e74c3c"))
        else:
            state = Text("⚫ 已禁用", style=Style(color="#95a5a6"))
        table.add_row(name, state, str(s["tool_count"]), s.get("description", "-"))

    return table


# ── 配置向导 ───────────────────────────────────────────

# 服务商预设（名称 → endpoint, model, 注册链接, 说明）
PROVIDER_PRESETS = {
    "1": {
        "name": "DeepSeek",
        "endpoint": "https://api.deepseek.com/v1",
        "models": "deepseek-chat, deepseek-reasoner",
        "default_model": "deepseek-chat",
        "key_url": "https://platform.deepseek.com/api_keys",
        "desc": "国产高性价比，新用户送500万tokens，推荐首选",
        "note": "需要注册获取 API Key → platform.deepseek.com",
    },
    "2": {
        "name": "阿里百炼 (Qwen)",
        "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": "qwen-turbo, qwen-plus, qwen-max",
        "default_model": "qwen-turbo",
        "key_url": "https://bailian.console.aliyun.com/",
        "desc": "阿里云旗下，中文能力强，有免费额度",
        "note": "需要阿里云账号 → bailian.console.aliyun.com",
    },
    "3": {
        "name": "Moonshot (Kimi)",
        "endpoint": "https://api.moonshot.cn/v1",
        "models": "moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k",
        "default_model": "moonshot-v1-8k",
        "key_url": "https://platform.moonshot.cn/",
        "desc": "长文本处理出色，适合文档分析",
        "note": "需要注册获取 Key → platform.moonshot.cn",
    },
    "4": {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1",
        "models": "gpt-4o, gpt-4o-mini, gpt-4-turbo",
        "default_model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
        "desc": "综合能力最强，价格较高",
        "note": "需要 OpenAI 账号 → platform.openai.com",
    },
    "5": {
        "name": "本地 Ollama（免费）",
        "endpoint": "http://localhost:11434/v1",
        "models": "qwen2:7b, llama3:8b, mistral:7b, deepseek-r1:8b",
        "default_model": "qwen2:7b",
        "key_url": None,
        "desc": "完全免费、离线可用、无需注册",
        "note": "需先安装 Ollama + 拉取模型 → ollama.com/download",
    },
}


def _render_provider_table() -> Table:
    """渲染服务商选择表。"""
    table = Table(
        title="选择 AI 服务商",
        border_style=Style(color="#c15f3f"),
        header_style=Style(color="#2f241c", bold=True),
    )
    table.add_column("按键", style=Style(color="#c15f3f", bold=True), width=6)
    table.add_column("服务商", style="bold", width=20)
    table.add_column("说明", width=40)
    table.add_column("价格", width=12)

    prices = {
        "1": "≈ ¥1/M tokens",
        "2": "≈ ¥2/M tokens",
        "3": "≈ ¥12/M tokens",
        "4": "≈ $5/M tokens",
        "5": "免费",
    }
    for key, p in PROVIDER_PRESETS.items():
        table.add_row(key, p["name"], p["desc"], prices.get(key, "-"))
    table.add_row("0", "[italic]自定义输入[/italic]", "自行输入任意 OpenAI 兼容的 API 地址", "-")

    return table


def config_wizard() -> dict:
    """两步配置向导：1) 选服务商 → 2) 确认/自定义参数。"""
    # ── 第一步：选择服务商 ───────────────────────
    console.print(Rule("第一步：选择 AI 服务商", style=STYLE_INFO))
    console.print(
        "  选择一个 AI 服务商来驱动 Agent 的思考能力。\n"
        "  如果不确定，选 [bold]1 - DeepSeek[/bold]（性价比最高，注册即用）\n"
        "  如果本地有 Ollama，选 [bold]5[/bold]（完全免费离线）\n",
        style=STYLE_INFO,
    )
    console.print(_render_provider_table())

    choice = Prompt.ask(
        "\n  [bold]请输入数字选择[/bold]",
        choices=["0", "1", "2", "3", "4", "5"],
        default="1",
    )

    # ── 第二步：确认并自定义参数 ────────────────
    preset = PROVIDER_PRESETS.get(choice)
    is_custom = (choice == "0")

    if is_custom:
        console.print(Rule("第二步：自定义 API 参数", style=STYLE_INFO))
        console.print(
            "  手动输入 API 地址。只要是 OpenAI 兼容的接口都可以用。\n"
            "  例如任何兼容 OpenAI SDK 的自部署模型服务（vLLM/OpenRouter 等）。\n",
            style=STYLE_INFO,
        )
        default_endpoint = "https://api.deepseek.com/v1"
        default_model = "deepseek-chat"
        preset_name = "自定义"
    else:
        console.print(Rule(f"第二步：配置 {preset['name']}", style=STYLE_INFO))
        console.print(f"  {preset['desc']}", style=STYLE_INFO)
        if preset.get("note"):
            console.print(f"  ⓘ {preset['note']}", style=Style(color="#9a887c"))
        default_endpoint = preset["endpoint"]
        default_model = preset["default_model"]
        preset_name = preset["name"]

    # Endpoint
    console.print()
    endp_help = Text(
        f"\n  API Endpoint = 模型的网络地址，决定了 Agent '找谁思考'。\n"
        f"  预设值适用于 {preset_name}，也可以自行修改为其他兼容 OpenAI 的地址。",
        style=Style(color="#9a887c"),
    )
    console.print(endp_help)

    base_url = Prompt.ask(
        "  [bold]API Endpoint[/bold]（直接回车使用预设，或自行输入）",
        default=default_endpoint,
    )

    # 模型名称
    if not is_custom:
        console.print(f"\n  {preset_name} 常用模型: [bold]{preset['models']}[/bold]", style=STYLE_INFO)

    model_help = Text(
        "\n  模型名称 = 具体用哪个模型来推理。不同模型能力/速度/价格不同。\n"
        "  可以填预设之外的其他模型名，只要是该 Endpoint 支持的即可。",
        style=Style(color="#9a887c"),
    )
    console.print(model_help)

    model_name = Prompt.ask(
        "  [bold]模型名称[/bold]（直接回车使用预设，或自行输入）",
        default=default_model,
    )

    # API Key
    console.print()
    if is_custom:
        key_help = Text(
            "\n  API Key = 你的身份凭证。大多数在线服务需要 Key 才能调用。\n"
            "  本地模型（Ollama）可以留空。",
            style=Style(color="#9a887c"),
        )
    elif preset.get("key_url"):
        key_help = Text(
            f"\n  API Key = {preset_name} 的身份凭证。\n"
            f"  获取地址: {preset['key_url']}\n"
            f"  粘贴时内容不会显示在屏幕上（隐私保护）。",
            style=Style(color="#9a887c"),
        )
    else:
        key_help = Text(
            "\n  API Key = 本地模型无需 Key，直接回车跳过。",
            style=Style(color="#9a887c"),
        )
    console.print(key_help)

    api_key = Prompt.ask(
        "  [bold]API Key[/bold]（本地模型可留空，输入时不回显）",
        password=True,
    )
    if not api_key or not api_key.strip():
        api_key = "sk-dummy-local-key"

    config = {"base_url": base_url, "api_key": api_key, "model_name": model_name}

    # ── 第三步：连接测试 ────────────────────────
    console.print()
    test = Confirm.ask(
        "  [bold]是否测试连接？[/bold]（发送一条测试请求验证配置是否正确）",
        default=True,
    )
    if test:
        from llm_client import LLMClient
        console.print()
        with console.status(f"[bold]正在连接 {preset_name}...[/bold]", spinner="dots"):
            try:
                test_llm = LLMClient(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    model=config["model_name"],
                )
                test_resp = test_llm.chat([
                    {"role": "user", "content": "回复 OK 即可。"}
                ], stream=False)
                console.print(
                    Panel(
                        f"连接成功！模型回复: {test_resp[:120]}",
                        border_style=Style(color="#27ae60"),
                        title="[bold]测试通过[/bold]",
                    )
                )
            except Exception as e:
                err_msg = str(e)
                # 提取关键信息
                if "401" in err_msg or "authentication" in err_msg.lower():
                    hint = "\n💡 API Key 可能无效，请检查 Key 是否正确。"
                elif "Connection" in err_msg or "refused" in err_msg.lower():
                    hint = "\n💡 无法连接服务器，请检查 Endpoint 地址是否正确。"
                elif "404" in err_msg:
                    hint = f"\n💡 模型 '{config['model_name']}' 可能不存在，请检查模型名称。"
                else:
                    hint = ""
                console.print(
                    Panel(
                        f"连接失败: {err_msg[:300]}{hint}",
                        border_style=Style(color="#e74c3c"),
                        title="[bold]测试失败[/bold]",
                    )
                )
                if not Confirm.ask(
                    "\n  [bold]连接失败，是否仍然使用当前配置继续？[/bold]",
                    default=True,
                ):
                    console.print("  已取消。请重新运行 python tui.py 配置。", style=STYLE_ERROR)
                    sys.exit(0)

    return config


def mcp_config_wizard() -> List[MCPServerConfig]:
    """交互式添加 MCP 服务器。"""
    configs = []
    console.print(Rule("MCP 服务器配置", style=STYLE_MCP))
    console.print("  MCP 服务器可以为 Agent 提供额外的工具能力（如文件系统访问、数据库查询等）。",
                   style=STYLE_INFO)

    while True:
        add = Confirm.ask("  添加 MCP 服务器？", default=False)
        if not add:
            break

        name = Prompt.ask("    服务器名称", default="my-server")
        command = Prompt.ask("    启动命令", default="npx")
        args_str = Prompt.ask("    命令行参数（用空格分隔）", default="")
        args = args_str.split() if args_str.strip() else []
        description = Prompt.ask("    描述（可选）", default="")
        configs.append(MCPServerConfig(
            name=name, command=command, args=args, description=description
        ))
        console.print(f"    ✅ 已添加 MCP 服务器: {name}", style=STYLE_INFO)

    return configs


# ── 主循环 ─────────────────────────────────────────────

class TUISession:
    """管理一个 TUI 会话的完整状态。"""

    def __init__(self, llm_config: dict, mcp_manager: Optional[MCPManager] = None):
        self.session_id = str(uuid.uuid4())
        self.session_name = "TUI-01"
        self.mcp_manager = mcp_manager
        self.llm_config = dict(llm_config)

        # 初始化 LLM
        self.llm = LLMClient(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            model=llm_config["model_name"],
        )

        # 初始化 Shell Agent（自然语言→命令执行）
        self.shell_agent = ShellAgent(self.llm, cwd=str(PROJECT_ROOT))

        # 初始化 MCP（连接并发现工具）
        if self.mcp_manager:
            with console.status("[bold]正在连接 MCP 服务器...[/bold]", spinner="dots"):
                self.mcp_manager.connect_all()

        # 初始化 Agent 链
        self.executor = ExecutorAgent(self.llm, AGENTS_MD, mcp_manager=self.mcp_manager)
        self.planner = PlannerAgent(self.llm, self.executor)
        self.memory = MemoryManager()
        self.orchestrator = OrchestratorAgent(
            self.llm, self.planner, self.memory, mcp_manager=self.mcp_manager
        )

        self.history: List[Dict[str, str]] = []
        self.trace_buffer: List[str] = []

    def refresh_mcp_tools(self):
        """重新连接并刷新 MCP 工具。"""
        if not self.mcp_manager:
            return 0
        self.mcp_manager.connect_all()
        count = self.executor._refresh_mcp_tools()
        self.executor.tools_desc = self.executor._build_tools_desc()
        self.executor.system_prompt = self.executor._build_prompt()
        return count

    def process_message(self, message: str):
        """流式处理用户消息，yield 各种 AgentEvent。"""
        self.trace_buffer = []
        msg = AgentMessage(
            sender="User",
            receiver="Orchestrator",
            content=message,
            metadata={"session_id": self.session_id, "history": self.history},
        )
        for event in self.orchestrator.process_stream(msg):
            if event.type == "trace":
                self.trace_buffer.append(event.data)
                yield event
            elif event.type == "thought":
                yield event
            elif event.type == "tool_call":
                yield event
            elif event.type == "tool_result":
                yield event
            elif event.type == "final":
                yield event

    def clear_memory(self):
        self.memory.clear(self.session_id)
        self.history = []


# ── 全局会话管理 ───────────────────────────────────────

ALL_TUI_SESSIONS: Dict[str, TUISession] = {}
TUI_SESSION_COUNTER = 0


def create_tui_session(llm_config: dict, mcp_manager: Optional[MCPManager] = None) -> TUISession:
    global TUI_SESSION_COUNTER
    TUI_SESSION_COUNTER += 1
    session = TUISession(llm_config, mcp_manager)
    session.session_name = f"TUI-{TUI_SESSION_COUNTER:02d}"
    ALL_TUI_SESSIONS[session.session_id] = session
    return session


# ── 命令处理 ───────────────────────────────────────────

def handle_command(cmd: str, session: TUISession, mcp_manager: MCPManager) -> Optional[str]:
    """处理内置命令，返回 None 表示不是命令（继续对话），返回字符串表示已处理。"""
    cmd_lower = cmd.strip().lower()

    if cmd_lower == "/quit" or cmd_lower == "/exit":
        return "EXIT"

    if cmd_lower == "/help":
        console.print(Panel(
            Text(
                "可用命令:\n"
                "  /help       - 显示此帮助\n"
                "  /clear      - 清空当前会话记忆\n"
                "  /mcp        - 查看 MCP 服务器状态\n"
                "  /mcp-reload - 重新连接 MCP 并刷新工具\n"
                "  /mcp-add    - 交互式添加 MCP 服务器\n"
                "  /shell      - Shell Agent 模式：自然语言→命令执行\n"
                "  /shell-rules- 查看 Shell Agent 安全规则列表\n"
                "  /sessions   - 列出所有会话\n"
                "  /new        - 创建新会话\n"
                "  /switch N   - 切换到会话 N\n"
                "  /quit       - 退出程序\n"
                "\n也可直接输入问题开始对话。",
                style=STYLE_INFO,
            ),
            title="帮助",
            border_style=Style(color="#c15f3f"),
        ))
        return "__HANDLED__"

    if cmd_lower == "/clear":
        session.clear_memory()
        console.print(Panel("✅ 记忆已清空", border_style=Style(color="#27ae60")))
        return "__HANDLED__"

    if cmd_lower == "/mcp":
        if mcp_manager:
            table = render_mcp_status_table(mcp_manager)
            console.print(table)
        else:
            console.print("⚠️ 未配置 MCP 管理器。", style=STYLE_ERROR)
        return "__HANDLED__"

    if cmd_lower == "/mcp-reload":
        if mcp_manager:
            with console.status("[bold]重新连接 MCP 服务器...[/bold]", spinner="dots"):
                count = session.refresh_mcp_tools()
            console.print(Panel(
                f"✅ MCP 工具已刷新，共注册 {count} 个远程工具。",
                border_style=Style(color="#27ae60"),
            ))
            if count > 0:
                table = render_mcp_status_table(mcp_manager)
                console.print(table)
        else:
            console.print("⚠️ 未配置 MCP 管理器。", style=STYLE_ERROR)
        return "__HANDLED__"

    if cmd_lower == "/mcp-add":
        configs = mcp_config_wizard()
        for cfg in configs:
            mcp_manager.add_server(cfg)
        if configs:
            mcp_manager.save_configs(str(DEFAULT_CONFIG_PATH))
            with console.status("[bold]连接新服务器...[/bold]", spinner="dots"):
                session.refresh_mcp_tools()
            console.print(f"✅ 已添加 {len(configs)} 个 MCP 服务器。", style=STYLE_INFO)
        return "__HANDLED__"

    if cmd_lower == "/sessions":
        table = Table(title="会话列表", border_style=Style(color="#c15f3f"))
        table.add_column("#", style="bold")
        table.add_column("ID")
        table.add_column("名称")
        for i, (sid, sess) in enumerate(ALL_TUI_SESSIONS.items()):
            marker = " ← 当前" if sid == session.session_id else ""
            table.add_row(str(i + 1), sid[:8] + "...", sess.session_name + marker)
        console.print(table)
        return "__HANDLED__"

    if cmd_lower == "/new":
        new_session = TUISession(session.llm_config, mcp_manager)
        ALL_TUI_SESSIONS[new_session.session_id] = new_session
        TUI_SESSION_COUNTER_REF = len(ALL_TUI_SESSIONS)
        new_session.session_name = f"TUI-{TUI_SESSION_COUNTER_REF:02d}"
        console.print(f"✅ 已创建新会话: {new_session.session_name}", style=STYLE_INFO)
        console.print(f"   使用 /switch {TUI_SESSION_COUNTER_REF} 切换到该会话", style=STYLE_INFO)
        return "__HANDLED__"

    # ── Shell Agent ───────────────────────────────
    if cmd_lower == "/shell-rules" or cmd_lower == "/safety-rules":
        rules = SafetyRuleEngine().list_rules()
        table = Table(title="Shell Agent 安全规则", border_style=Style(color="#e74c3c"))
        table.add_column("#", style="dim")
        table.add_column("风险等级")
        table.add_column("规则描述")
        for i, rule in enumerate(rules, 1):
            parts = rule.split(" | ", 1)
            risk = Text(parts[0].strip(), style=Style(
                color="#e74c3c" if "high" in parts[0].strip() else "#e67e22"
                if "medium" in parts[0].strip() else "#27ae60"
            ))
            desc = parts[1].strip() if len(parts) > 1 else rule
            table.add_row(str(i), risk, desc)
        console.print(table)
        return "__HANDLED__"

    shell_match = re.match(r"^/shell\s+(.+)", cmd.strip(), re.IGNORECASE)
    if shell_match:
        nl_input = shell_match.group(1).strip()
        console.print(Rule("Shell Agent（只读演示模式）", style=Style(color="#4a90d9")))
        console.print(f"  [bold]自然语言:[/bold] {nl_input}", style=STYLE_INFO)

        # 第一步: LLM 结构化
        with console.status("[bold]Shell Agent 分析中...[/bold]", spinner="dots"):
            cmd_info, safety_verdict, needs_confirm, result = session.shell_agent.run_pipeline(nl_input)

        # 显示 LLM 生成的命令信息
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column(style="dim", width=12)
        info_table.add_column()
        info_table.add_row("意图", Text(cmd_info.get("intent", "?"), style=Style(
            color="#27ae60" if cmd_info.get("intent") == "run_command" else "#e67e22")))
        info_table.add_row("风险等级", Text(cmd_info.get("risk_level", "?").upper(), style=Style(
            color="#e74c3c" if cmd_info.get("risk_level") == "high" else
            "#e67e22" if cmd_info.get("risk_level") == "medium" else "#27ae60"
        )))
        info_table.add_row("命令", Text(f"`{cmd_info.get('command', '(无)')}`", style=Style(color="#4a90d9")))
        info_table.add_row("理由", Text(cmd_info.get("reason", ""), style=STYLE_INFO))
        console.print(info_table)

        # 处理不同意图
        if cmd_info.get("intent") == "refuse":
            console.print(Panel(
                Text(f"⛔ LLM 拒绝执行: {cmd_info.get('reason', '')}", style=STYLE_ERROR),
                border_style=Style(color="#e74c3c"),
            ))
            return "__HANDLED__"

        if cmd_info.get("intent") == "ask_clarification":
            console.print(Panel(
                Text(f"❓ 需要澄清: {cmd_info.get('reason', '')}", style=STYLE_INFO),
                border_style=Style(color="#e67e22"),
            ))
            return "__HANDLED__"

        # 安全裁决详情
        if safety_verdict.matched_rules:
            console.print(Text("  ⚠️ 本地规则引擎匹配:", style=Style(color="#e67e22")))
            for r in safety_verdict.matched_rules:
                console.print(Text(f"    • {r}", style=Style(color="#e67e22")))

        # 需要用户确认？
        if needs_confirm:
            console.print()
            console.print(Panel(
                Text(
                    f"命令: {cmd_info['command']}\n"
                    f"风险: {cmd_info.get('risk_level', '?').upper()}\n"
                    f"原因: {cmd_info.get('reason', '')}\n\n"
                    f"{'附加拦截: ' + safety_verdict.reason if safety_verdict.reason else ''}",
                    style=Style(color="#e74c3c"),
                ),
                title="⚠️ 需要确认",
                border_style=Style(color="#e74c3c"),
            ))
            if Confirm.ask("  [bold red]是否继续执行此命令？[/bold red]", default=False):
                console.print()
                result = session.shell_agent.execute_pending()
                if result is None:
                    return "__HANDLED__"
            else:
                console.print("  ❌ 已取消。", style=STYLE_INFO)
                return "__HANDLED__"

        # 显示执行结果
        if result:
            console.print(Rule("执行结果", style=Style(color="#27ae60" if result.success else "#e74c3c")))
            if result.stdout:
                md = result.stdout[:3000]
                try:
                    console.print(Panel(md, border_style=Style(color="#4a90d9"), padding=(1, 1)))
                except Exception:
                    console.print(Text(md))
            if result.stderr:
                console.print(Panel(
                    Text(result.stderr[:1000], style=STYLE_ERROR),
                    title="stderr",
                    border_style=Style(color="#e74c3c"),
                ))
            exit_style = Style(color="#27ae60") if result.exit_code == 0 else Style(color="#e74c3c")
            console.print(Text(f"  退出码: {result.exit_code}  |  已确认: {'是' if result.was_confirmed else '否'}", style=exit_style))
        else:
            # 这是低风险直接执行的情况，result 已在 run_pipeline 中返回
            pass

        return "__HANDLED__"

    match = re.match(r"^/switch\s+(\d+)$", cmd.strip(), re.IGNORECASE)
    if match:
        idx = int(match.group(1)) - 1
        sessions_list = list(ALL_TUI_SESSIONS.values())
        if 0 <= idx < len(sessions_list):
            new_active = sessions_list[idx]
            console.print(
                f"🔄 请手动设置活动会话为: {new_active.session_name} (sid={new_active.session_id[:8]}...)",
                style=STYLE_INFO,
            )
        else:
            console.print("⚠️ 无效的会话编号。", style=STYLE_ERROR)
        return "__HANDLED__"

    return None  # 不是命令


# ── 主入口 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Agent Lite - TUI 终端界面")
    parser.add_argument("--api-key", type=str, default=None, help="LLM API Key")
    parser.add_argument("--base-url", type=str, default=None, help="LLM API Endpoint")
    parser.add_argument("--model", type=str, default=None, help="模型名称")
    parser.add_argument("--mcp-config", type=str, default=None,
                        help="MCP 服务器配置文件路径 (JSON)")
    parser.add_argument("--skip-config", action="store_true",
                        help="跳过交互式配置（使用默认值）")
    args = parser.parse_args()

    # ── 加载 MCP 配置 ───────────────────────────────
    mcp_config_path = args.mcp_config or str(DEFAULT_CONFIG_PATH)
    mcp_manager = MCPManager.load_configs(mcp_config_path)

    # ── LLM 配置 ────────────────────────────────────
    if args.skip_config:
        llm_config = {
            "base_url": args.base_url or "https://api.deepseek.com/v1",
            "api_key": args.api_key or "sk-dummy-local-key",
            "model_name": args.model or "deepseek-chat",
        }
    else:
        # 启动动画
        render_banner()

        # 如果有 MCP 配置则显示
        if mcp_manager.get_configs():
            console.print(Rule("已加载 MCP 配置", style=STYLE_MCP))
            table = render_mcp_status_table(mcp_manager)
            console.print(table)

        # 交互式配置
        llm_config = {
            "base_url": args.base_url or "https://api.deepseek.com/v1",
            "api_key": args.api_key or "sk-dummy-local-key",
            "model_name": args.model or "deepseek-chat",
        }

        # 如果命令行没有提供完整配置，启动向导
        if not args.api_key or not args.model:
            user_config = config_wizard()
            llm_config.update({k: v for k, v in user_config.items() if v})
            # 命令行参数优先级更高
            if args.base_url:
                llm_config["base_url"] = args.base_url
            if args.api_key:
                llm_config["api_key"] = args.api_key
            if args.model:
                llm_config["model_name"] = args.model

    # ── 创建会话 ────────────────────────────────────
    console.print(Rule("初始化 Agent", style=STYLE_INFO))
    with console.status("[bold]正在启动 Agent...[/bold]", spinner="dots"):
        try:
            # 连接 MCP 服务器
            if mcp_manager.get_configs():
                with console.status("[bold]正在连接 MCP 服务器...[/bold]", spinner="dots"):
                    mcp_manager.connect_all()

            session = create_tui_session(llm_config, mcp_manager)
        except Exception as e:
            console.print(f"❌ 初始化失败: {str(e)}", style=STYLE_ERROR)
            sys.exit(1)

    console.print(Panel(
        f"✅ Agent 已就绪: {session.session_name}\n"
        f"   模型: {llm_config['model_name']}\n"
        f"   端点: {llm_config['base_url']}\n"
        f"   MCP 服务器: {len(mcp_manager.get_configs())} 个配置, "
        f"{sum(1 for s in mcp_manager.get_server_status().values() if s['connected'])} 个已连接",
        border_style=Style(color="#27ae60"),
    ))

    if mcp_manager.get_configs():
        table = render_mcp_status_table(mcp_manager)
        console.print(table)

    console.print(Rule("开始对话", style=STYLE_INFO))
    console.print("  输入 /help 查看命令 | 直接输入问题开始对话\n", style=STYLE_INFO)

    # ── 主对话循环 ──────────────────────────────────
    while True:
        try:
            user_input = Prompt.ask("\n[bold #c15f3f]▸[/bold #c15f3f]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n👋 再见~", style=STYLE_INFO)
            break

        if not user_input or not user_input.strip():
            continue

        # 命令处理
        cmd_result = handle_command(user_input, session, mcp_manager)
        if cmd_result == "EXIT":
            console.print("👋 再见~", style=STYLE_INFO)
            break
        if cmd_result == "__HANDLED__":
            continue

        # ── 流式对话处理 ──────────────────────────
        console.print()  # 空行分隔

        final_output = ""
        trace_displayed = set()

        try:
            for event in session.process_message(user_input.strip()):
                if event.type == "trace":
                    line = event.data
                    if line not in trace_displayed:
                        trace_displayed.add(line)
                        console.print(_format_trace_line(line))
                elif event.type == "thought":
                    console.print(Panel(
                        Text(event.data, style=STYLE_THOUGHT),
                        title="💭 思考",
                        border_style=Style(color="#5a7a4a"),
                        padding=(0, 1),
                    ))
                elif event.type == "tool_call":
                    info = event.data
                    console.print(Text(
                        f"🔧 调用: {info['action']}({json.dumps(info['input'], ensure_ascii=False)})",
                        style=STYLE_TOOL_CALL,
                    ))
                elif event.type == "tool_result":
                    console.print(Panel(
                        Text(str(event.data)[:1000], style=STYLE_TOOL_RESULT),
                        title="👁️ 工具返回",
                        border_style=Style(color="#6b5a4e"),
                        padding=(0, 1),
                    ))
                elif event.type == "final":
                    final_output = event.data
        except Exception as e:
            console.print(Panel(
                Text(f"对话处理异常: {str(e)}", style=STYLE_ERROR),
                title="❌ 错误",
                border_style=Style(color="#cc3333"),
            ))
            final_output = f"[处理失败] {str(e)}"

        # ── 渲染最终回复 ───────────────────────────
        if final_output:
            console.print(Rule("回复", style=Style(color="#c15f3f")))
            try:
                md = Markdown(final_output, style=STYLE_ASSISTANT)
                console.print(Panel(md, border_style=Style(color="#c15f3f"), padding=(1, 2)))
            except Exception:
                console.print(Panel(
                    Text(final_output, style=STYLE_ASSISTANT),
                    border_style=Style(color="#c15f3f"),
                    padding=(1, 2),
                ))

        # 更新历史
        session.history.append({"role": "user", "content": user_input.strip()})
        session.history.append({"role": "assistant", "content": final_output})

    # ── 清理 ───────────────────────────────────────
    if mcp_manager:
        mcp_manager.disconnect_all()
        mcp_manager.save_configs(mcp_config_path)


if __name__ == "__main__":
    main()
