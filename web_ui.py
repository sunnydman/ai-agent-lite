import os
import re
import time
import uuid
import json
import threading
import shlex

import gradio as gr
from llm_client import LLMClient
from agents import OrchestratorAgent, PlannerAgent, ExecutorAgent, AgentMessage
from memory import MemoryManager
from mcp_client import MCPManager, MCPServerConfig

try:
    with open("AGENTS.md", "r", encoding="utf-8") as f:
        AGENTS_MD = f.read()
except FileNotFoundError:
    AGENTS_MD = "未找到 AGENTS.md 文件。"

memory_mgr = MemoryManager()

# 🌐 全局 MCP 管理器
MCP_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_servers.json")
mcp_manager = MCPManager.load_configs(MCP_CONFIG_PATH)

# 🌐 全局会话管理器
ALL_SESSIONS = {}
SESSION_COUNTER = 0

# 简洁工作台风格 CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&family=Noto+Serif+SC:wght@600;700&display=swap');

body {
    background: #f7f1e8 !important;
    color: #2f241c !important;
    margin: 0 !important;
    overflow-x: hidden !important;
}

.gradio-container {
    width: 100vw !important;
    max-width: none !important;
    min-height: 100vh !important;
    margin: 0 !important;
    padding: 24px 36px !important;
    background: transparent !important;
    font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    box-sizing: border-box !important;
    position: relative !important;
}

.gradio-container::before {
    content: "" !important;
    position: fixed !important;
    inset: 18px 22px !important;
    pointer-events: none !important;
    border: 1px solid rgba(193, 95, 63, 0.16) !important;
    border-radius: 24px !important;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.38) !important;
    z-index: 0 !important;
}

.gradio-container > * {
    position: relative !important;
    z-index: 1 !important;
}

.gradio-container a,
.gradio-container a:visited {
    color: #8f4328 !important;
}

.gradio-container label,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container .block-label,
.gradio-container .block-info {
    color: #756458 !important;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .input-container {
    background: #fffaf3 !important;
    border-color: rgba(193, 95, 63, 0.18) !important;
}

.gradio-container .wrap-inner,
.gradio-container .secondary-wrap {
    background: #fffdf9 !important;
    border-color: rgba(193, 95, 63, 0.18) !important;
}

.gradio-container details,
.gradio-container .accordion {
    background: #fffaf3 !important;
    border: 1px solid rgba(193, 95, 63, 0.2) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 30px rgba(91, 61, 38, 0.05) !important;
}

.gradio-container summary,
.gradio-container .accordion .label-wrap {
    background: #fbf0e6 !important;
    color: #2f241c !important;
    border-color: rgba(193, 95, 63, 0.16) !important;
}

.gradio-container hr {
    border-color: rgba(193, 95, 63, 0.16) !important;
}

.gradio-container [role="listbox"],
.gradio-container [data-testid="dropdown-options"],
.gradio-container .options {
    background: #fffaf3 !important;
    border: 1px solid rgba(193, 95, 63, 0.2) !important;
    box-shadow: 0 16px 32px rgba(91, 61, 38, 0.1) !important;
}

.gradio-container [role="option"] {
    background: #fffaf3 !important;
    color: #2f241c !important;
}

.gradio-container [role="option"]:hover,
.gradio-container [aria-selected="true"] {
    background: #fbf0e6 !important;
    color: #8f4328 !important;
}

.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .output-markdown {
    color: #2f241c !important;
}

.gradio-container .prose p,
.gradio-container .markdown p,
.gradio-container .output-markdown p {
    color: #756458 !important;
}

.gradio-container .main.fillable,
.gradio-container main.contain,
.gradio-container .wrap {
    width: 100% !important;
    max-width: none !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

.gradio-container .column {
    gap: 12px !important;
}

.block.app-title h1,
.block.app-title h2,
.block.app-title h3,
.block.app-title p {
    margin: 0 !important;
}

.block.app-title {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding-left: 4px !important;
    color: #2f241c !important;
    position: relative !important;
}

.block.app-title::before {
    content: "" !important;
    display: block !important;
    width: 132px !important;
    height: 3px !important;
    margin-bottom: 14px !important;
    border-radius: 999px !important;
    background: #c15f3f !important;
}

.prose.app-title,
.prose.chat-header,
.prose.status-bar,
.prose.sidebar-title,
.prose.init-status {
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.prose.app-title::before {
    content: none !important;
    display: none !important;
}

.block.app-title h1 {
    font-family: 'Noto Serif SC', Georgia, serif !important;
    font-size: 34px !important;
    line-height: 1.25 !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    color: #2f241c !important;
}

.block.app-title p {
    margin-top: 6px !important;
    color: #756458 !important;
    font-size: 14px !important;
}

.main-row {
    width: 100% !important;
    max-width: none !important;
    height: auto !important;
    min-height: 0 !important;
    margin: 0 !important;
    gap: 20px !important;
    align-items: stretch !important;
    display: flex !important;
    flex-wrap: nowrap !important;
}

.sidebar {
    flex: 0 0 280px !important;
    width: 280px !important;
    min-width: 280px !important;
    max-width: 280px !important;
    height: auto !important;
    background: #fffaf3 !important;
    border: 1px solid rgba(193, 95, 63, 0.22) !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 18px 44px rgba(91, 61, 38, 0.08) !important;
    box-sizing: border-box !important;
    backdrop-filter: blur(14px) !important;
    position: relative !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}

.sidebar::before {
    content: "" !important;
    position: absolute !important;
    inset: 0 auto 0 0 !important;
    width: 4px !important;
    background: #c15f3f !important;
    opacity: 1 !important;
}

.sidebar-title h3 {
    margin: 0 0 12px !important;
    color: #2f241c !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.sidebar-subtitle h3,
.sidebar-subtitle p {
    margin: 0 !important;
}

.sidebar-subtitle h3 {
    color: #2f241c !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}

.agent-select-card {
    width: 100% !important;
    min-height: 118px !important;
    padding: 14px 16px !important;
    border: 1px solid rgba(193, 95, 63, 0.18) !important;
    background: #fffdf9 !important;
    box-shadow: none !important;
}

.agent-select-card > div {
    height: 100% !important;
}

.agent-create-button {
    width: 100% !important;
    min-height: 60px !important;
    border-radius: 12px !important;
}

.chat-panel {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    height: auto !important;
    min-height: 0 !important;
    background: #fffaf3 !important;
    border: 1px solid rgba(193, 95, 63, 0.22) !important;
    border-radius: 20px !important;
    padding: 0 !important;
    overflow: hidden !important;
    box-shadow: 0 24px 62px rgba(91, 61, 38, 0.09) !important;
    backdrop-filter: blur(16px) !important;
    display: flex !important;
    flex-direction: column !important;
}

.block.chat-header {
    padding: 18px 22px 14px !important;
    border-bottom: 1px solid rgba(193, 95, 63, 0.16) !important;
    background: #fbf0e6 !important;
}

.block.chat-header h3,
.block.chat-header p {
    margin: 0 !important;
}

.block.chat-header h3 {
    font-family: 'Noto Serif SC', Georgia, serif !important;
    color: #2f241c !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.block.chat-header p {
    margin-top: 4px !important;
    color: #756458 !important;
    font-size: 13px !important;
}

.config-panel {
    width: 100% !important;
    max-width: none !important;
    margin: 0 0 12px !important;
    border: 1px solid rgba(193, 95, 63, 0.2) !important;
    border-radius: 18px !important;
    background: #fffaf3 !important;
    box-shadow: 0 14px 34px rgba(91, 61, 38, 0.06) !important;
    backdrop-filter: blur(14px) !important;
}

.config-panel > div {
    border-radius: 16px !important;
}

.init-status {
    color: #6b5a4e !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}

.mcp-status-card {
    padding: 8px 12px !important;
    border-radius: 10px !important;
    background: #fdf6f0 !important;
    border: 1px solid rgba(155, 89, 182, 0.18) !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
}

.mcp-server-row {
    padding: 6px 10px !important;
    margin: 2px 0 !important;
    border-radius: 8px !important;
    background: #fdf6f0 !important;
    border: 1px solid rgba(193, 95, 63, 0.12) !important;
}

button.primary,
button.secondary,
button {
    background: #fffaf3 !important;
    border: 1px solid rgba(193, 95, 63, 0.24) !important;
    color: #2f241c !important;
    min-height: 42px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    transition: background-color 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease !important;
}

button.primary {
    background: #c15f3f !important;
    border: 1px solid #c15f3f !important;
    color: #ffffff !important;
}

button.primary:hover {
    background: #a94f32 !important;
    border-color: #a94f32 !important;
    box-shadow: 0 12px 26px rgba(189, 91, 53, 0.22) !important;
}

button.secondary {
    background: #fffaf3 !important;
    border: 1px solid rgba(193, 95, 63, 0.28) !important;
    color: #2f241c !important;
}

button.secondary:hover {
    background: #fbf0e6 !important;
    border-color: #c15f3f !important;
}

button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(193, 95, 63, 0.14) !important;
}

textarea,
input,
select {
    background: #fffdf9 !important;
    color: #2f241c !important;
    border: 1px solid rgba(193, 95, 63, 0.28) !important;
    border-radius: 10px !important;
    font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 14px !important;
}

textarea::placeholder,
input::placeholder {
    color: #9a887c !important;
    opacity: 1 !important;
}

textarea:focus,
input:focus,
select:focus {
    border-color: #c15f3f !important;
    box-shadow: 0 0 0 3px rgba(193, 95, 63, 0.12) !important;
}

.chat-input-row {
    padding: 14px 18px !important;
    border-top: 1px solid rgba(193, 95, 63, 0.16) !important;
    background: #fffaf3 !important;
    align-items: stretch !important;
    gap: 12px !important;
    margin-top: 0 !important;
    display: flex !important;
    flex-wrap: nowrap !important;
}

.chat-input-row .form {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    width: auto !important;
}

.chat-input-row button {
    flex: 0 0 96px !important;
    width: 96px !important;
    min-width: 96px !important;
    max-width: 96px !important;
    min-height: 72px !important;
}

.chat-input-row textarea {
    min-height: 56px !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    resize: vertical !important;
}

.chat-panel > .block.flex {
    height: 420px !important;
    min-height: 420px !important;
}

.chat-panel .chatbot,
.chat-panel [data-testid="chatbot"],
.chat-panel .messages,
.chat-panel .message-wrap {
    background: #fffaf3 !important;
    border: 0 !important;
}

.message .message-bubble {
    padding: 13px 16px !important;
    border-radius: 16px !important;
    font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.65 !important;
    box-shadow: none !important;
}

.message.user .message-bubble {
    background: #8f4328 !important;
    color: #ffffff !important;
    border: 1px solid #8f4328 !important;
    max-width: 78% !important;
}

.message.bot .message-bubble {
    background: #fbf0e6 !important;
    border: 1px solid rgba(193, 95, 63, 0.16) !important;
    color: #2f241c !important;
}

.block.status-bar {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 10px 14px !important;
    border: 1px solid rgba(193, 95, 63, 0.2) !important;
    border-radius: 16px !important;
    background: #fffaf3 !important;
    color: #756458 !important;
    font-size: 13px !important;
    box-shadow: 0 12px 28px rgba(91, 61, 38, 0.05) !important;
    backdrop-filter: blur(12px) !important;
}

.prose.status-bar p {
    margin: 0 !important;
}

.message.bot pre, .message.bot code {
    background: #fffdf9 !important;
    border: 1px solid rgba(193, 95, 63, 0.16) !important;
    color: #4a3a30 !important;
    font-size: 13px !important;
    border-radius: 10px !important;
}

.gradio-container .toast,
.gradio-container .notification,
.gradio-container .toast-body {
    background: #fffaf3 !important;
    border-color: rgba(193, 95, 63, 0.24) !important;
    color: #2f241c !important;
}

footer { display: none !important; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d4ad9d; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #c15f3f; }

@media (max-width: 760px) {
    .gradio-container {
        padding: 16px 12px !important;
    }

    .main-row {
        width: 100% !important;
        height: auto !important;
        min-height: 0 !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
    }

    .sidebar {
        width: 100% !important;
        min-width: 0 !important;
        max-width: none !important;
        height: auto !important;
    }

    .chat-panel {
        min-height: 0 !important;
    }

    .chat-panel > .block.flex {
        height: 320px !important;
        min-height: 320px !important;
    }

    .chat-input-row {
        flex-direction: row !important;
    }
}
"""


def create_session(base_url, api_key, model_name, current_choices):
    global SESSION_COUNTER
    if not base_url:
        gr.Warning("⚠️ 请输入 API Endpoint")
        return gr.update(), gr.update(), [], None

    if not api_key: api_key = "sk-dummy-local-key"

    try:
        # 确保 MCP 已连接
        if mcp_manager.get_configs():
            mcp_manager.connect_all()

        llm = LLMClient(api_key=api_key, base_url=base_url, model=model_name)
        executor = ExecutorAgent(llm, AGENTS_MD, mcp_manager=mcp_manager)
        planner = PlannerAgent(llm, executor)
        orchestrator = OrchestratorAgent(llm, planner, memory_mgr, mcp_manager=mcp_manager)

        SESSION_COUNTER += 1
        sid = str(uuid.uuid4())
        name = f"Node-{SESSION_COUNTER:02d}"

        ALL_SESSIONS[sid] = {
            "name": name,
            "orchestrator": orchestrator,
            "history": []
        }

        choices = [(v["name"], k) for k, v in ALL_SESSIONS.items()]
        gr.Info(f"{name} 已上线")
        return gr.update(choices=choices, value=sid), gr.update(value=f"{name} 已连接"), [], sid
    except Exception as e:
        gr.Error(f"✖ 连接失败: {str(e)}")
        return gr.update(), gr.update(value=f"✖ ERROR: {str(e)}"), [], None


def switch_session(sid):
    if not sid or sid not in ALL_SESSIONS:
        return []
    return ALL_SESSIONS[sid]["history"]


def create_session_for_ui(base_url, api_key, model_name, current_choices):
    dropdown_update, status_update, history, sid = create_session(
        base_url, api_key, model_name, current_choices
    )
    secondary_update = dict(dropdown_update) if isinstance(dropdown_update, dict) else dropdown_update
    return dropdown_update, secondary_update, status_update, history, sid


def switch_session_for_ui(sid):
    return switch_session(sid), gr.update(value=sid), sid


def chat_fn(message, history, sid):
    history = history or []
    if not message or not message.strip():
        yield history, ""
        return

    if not sid or sid not in ALL_SESSIONS:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ 请先在左侧创建并选择一个 Agent 节点。"})
        yield history, ""
        return

    session = ALL_SESSIONS[sid]
    orchestrator = session["orchestrator"]
    history = history or []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ [PROCESSING...]"})
    yield history, ""

    msg = AgentMessage(sender="User", receiver="Orchestrator", content=message,
                       metadata={"session_id": sid})

    try:
        response = orchestrator.process(msg)
    except Exception as e:
        history[-1]["content"] = f"✖ [SYSTEM CRASH] : {str(e)}"
        yield history, ""
        return

    trace = response.metadata.get("trace", [])
    displayed_trace = ""

    for step in trace:
        displayed_trace += f"{step}\n"
        history[-1]["content"] = f"```bash\n[SYSTEM_TRACE]\n{displayed_trace}```"
        yield history, ""
        time.sleep(0.08)

    final_text = response.content
    displayed_text = ""
    sleep_time = 0.015 if len(final_text) < 200 else 0.005

    for char in final_text:
        displayed_text += char
        history[-1]["content"] = f"```bash\n[SYSTEM_TRACE]\n{displayed_trace}```\n\n**[OUTPUT]**\n{displayed_text}"
        yield history, ""
        time.sleep(sleep_time)

    # 同步到全局状态
    session["history"] = history


# ── MCP 管理函数 ───────────────────────────────────────

def parse_mcp_args(args_str):
    """Parse MCP command arguments while preserving quoted paths."""
    if not args_str or not args_str.strip():
        return []
    return [arg.strip("\"'") for arg in shlex.split(args_str, posix=False)]


def mcp_add_server(name, command, args_str, description):
    """添加 MCP 服务器配置。"""
    if not name or not command:
        return gr.update(), "⚠️ 请填写服务器名称和启动命令"

    try:
        args = parse_mcp_args(args_str)
    except ValueError as exc:
        return gr.update(), f"⚠️ 参数解析失败: {exc}"
    config = MCPServerConfig(name=name, command=command, args=args, description=description)
    mcp_manager.add_server(config)
    mcp_manager.save_configs(MCP_CONFIG_PATH)

    # 尝试连接
    try:
        mcp_manager.connect_all()
    except Exception:
        pass

    status_text = _build_mcp_status_text()
    return gr.update(value=""), status_text


def mcp_remove_server(server_name):
    """移除 MCP 服务器配置。"""
    if not server_name:
        return _build_mcp_status_text()
    mcp_manager.remove_server(server_name)
    mcp_manager.save_configs(MCP_CONFIG_PATH)
    return _build_mcp_status_text()


def mcp_reconnect():
    """重新连接所有 MCP 服务器并刷新工具。"""
    mcp_manager.connect_all()
    # 刷新所有现有 session 的工具
    for sid, session in ALL_SESSIONS.items():
        if hasattr(session["orchestrator"], "executor") or hasattr(session["orchestrator"].planner, "executor"):
            try:
                orchestrator = session["orchestrator"]
                # 直接到 executor
                exec_agent = orchestrator.planner.executor
                exec_agent._refresh_mcp_tools()
                exec_agent.tools_desc = exec_agent._build_tools_desc()
                exec_agent.system_prompt = exec_agent._build_prompt()
            except Exception:
                pass
    return _build_mcp_status_text()


def _build_mcp_status_text() -> str:
    """构建 MCP 状态文本。"""
    status = mcp_manager.get_server_status()
    if not status:
        return "📭 未配置 MCP 服务器\n\n可通过下方表单添加 MCP 服务器来扩展 Agent 能力。"
    lines = ["**MCP 服务器状态:**\n"]
    for sname, s in status.items():
        if s["connected"]:
            icon = "🟢"
            info = f"在线 · {s['tool_count']} 个工具"
        elif s["enabled"]:
            icon = "🔴"
            info = "离线"
        else:
            icon = "⚫"
            info = "已禁用"
        desc = f" — {s['description']}" if s.get("description") else ""
        lines.append(f"- {icon} **{sname}**: {info}{desc}")

    total_tools = sum(s["tool_count"] for s in status.values())
    lines.append(f"\n📊 总计 {len(status)} 个服务器, {total_tools} 个远程工具")
    return "\n".join(lines)


def get_status():
    mcp_count = len(mcp_manager.get_configs()) if mcp_manager else 0
    mcp_connected = sum(
        1 for s in mcp_manager.get_server_status().values() if s.get("connected")
    ) if mcp_manager else 0
    return (
        f"状态：在线 · 时间：{time.strftime('%H:%M:%S')} · "
        f"节点：{len(ALL_SESSIONS)} · "
        f"MCP：{mcp_connected}/{mcp_count} 已连接"
    )


with gr.Blocks() as demo:
    gr.Markdown(
        "# AI Agent Lite\n轻量多 Agent 工作台，支持规划、工具调用与长期记忆。",
        elem_classes="app-title",
    )
    timer = gr.Timer(5)

    # 隐藏的配置区
    with gr.Accordion("模型配置", open=False, elem_classes="config-panel"):
        with gr.Row():
            base_url_select = gr.Dropdown(
                choices=[
                    "https://api.deepseek.com/v1",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "https://api.moonshot.cn/v1",
                    "http://localhost:11434/v1",
                    "https://api.openai.com/v1"
                ],
                value="https://api.deepseek.com/v1",
                label="API Endpoint",
                allow_custom_value=True,
                scale=2
            )
            api_key_input = gr.Textbox(label="API Key（本地模型可留空）", type="password", scale=2)
        with gr.Row():
            model_select = gr.Dropdown(
                choices=["deepseek-chat", "qwen-turbo", "moonshot-v1-8k", "qwen2:7b", "gpt-4o"],
                value="deepseek-chat", label="模型名称", allow_custom_value=True, scale=2
            )
            init_btn = gr.Button("初始化节点", variant="primary", scale=1)
        init_status = gr.Markdown("", elem_classes="init-status")

    status_box = gr.Markdown(value=get_status(), elem_classes="status-bar")

    # 主布局：左侧边栏 + 右侧聊天
    with gr.Row(elem_classes="main-row"):
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### Agent 节点", elem_classes="sidebar-title")
            session_dropdown = gr.Dropdown(
                label="选择 Agent",
                choices=[],
                interactive=True,
                elem_classes="agent-select-card"
            )
            new_session_btn = gr.Button("创建新 Agent", variant="secondary", elem_classes="agent-create-button")
            gr.Markdown("### 当前 Agent", elem_classes="sidebar-subtitle")
            session_dropdown_bottom = gr.Dropdown(
                label="快速切换",
                choices=[],
                interactive=True,
                elem_classes="agent-select-card"
            )
            gr.Markdown("---")

            # ── MCP 管理区 ───────────────────────────
            with gr.Accordion("🔌 MCP 服务器管理", open=False):
                mcp_status_md = gr.Markdown(
                    value=_build_mcp_status_text(),
                    elem_classes="init-status"
                )
                with gr.Row():
                    mcp_reconnect_btn = gr.Button("🔄 重连 MCP", variant="secondary", size="sm")
                with gr.Row():
                    mcp_name = gr.Textbox(label="服务器名称", placeholder="如 filesystem", scale=1)
                    mcp_command = gr.Textbox(label="启动命令", placeholder="如 npx", scale=1)
                mcp_args = gr.Textbox(label="命令行参数（空格分隔）",
                                       placeholder='如 -y @modelcontextprotocol/server-filesystem /path')
                mcp_desc = gr.Textbox(label="描述（可选）", placeholder="简要描述该服务器功能")
                with gr.Row():
                    mcp_add_btn = gr.Button("➕ 添加 MCP 服务器", variant="primary", size="sm")
                mcp_remove_dropdown = gr.Dropdown(
                    label="移除服务器",
                    choices=[],
                    interactive=True,
                )
                mcp_remove_btn = gr.Button("🗑 移除选中服务器", variant="secondary", size="sm")

            gr.Markdown("---")
            gr.Markdown("先配置模型，再创建节点。创建后即可在右侧开始对话。",
                        elem_classes="init-status")

        with gr.Column(scale=4, elem_classes="chat-panel"):
            gr.Markdown("### 对话\n输入任务后，Agent 会展示推理轨迹和最终输出。", elem_classes="chat-header")
            active_sid = gr.State(None)
            chatbot = gr.Chatbot(height=420, show_label=False)
            with gr.Row(elem_classes="chat-input-row"):
                msg = gr.Textbox(
                    placeholder="输入问题或任务，例如：计算 (2^10 + 3^5) * 7",
                    show_label=False,
                    lines=2,
                    scale=8,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            # 事件绑定
            msg.submit(chat_fn, [msg, chatbot, active_sid], [chatbot, msg])
            send_btn.click(chat_fn, [msg, chatbot, active_sid], [chatbot, msg])

            # 切换节点时，加载对应的 history
            session_dropdown.change(
                switch_session_for_ui,
                [session_dropdown],
                [chatbot, session_dropdown_bottom, active_sid]
            )
            session_dropdown_bottom.change(
                switch_session_for_ui,
                [session_dropdown_bottom],
                [chatbot, session_dropdown, active_sid]
            )

    # 创建节点事件
    new_session_btn.click(
        create_session_for_ui,
        [base_url_select, api_key_input, model_select, session_dropdown],
        [session_dropdown, session_dropdown_bottom, init_status, chatbot, active_sid]
    )

    # 初始化按钮复用创建逻辑
    init_btn.click(
        create_session_for_ui,
        [base_url_select, api_key_input, model_select, session_dropdown],
        [session_dropdown, session_dropdown_bottom, init_status, chatbot, active_sid]
    )

    # ── MCP 事件绑定 ───────────────────────────────
    def _refresh_mcp_dropdown():
        configs = mcp_manager.get_configs()
        return gr.update(choices=[c.name for c in configs], value=None)

    mcp_add_btn.click(
        mcp_add_server,
        [mcp_name, mcp_command, mcp_args, mcp_desc],
        [mcp_name, mcp_status_md]
    ).then(
        _refresh_mcp_dropdown, None, [mcp_remove_dropdown]
    )

    mcp_remove_btn.click(
        mcp_remove_server,
        [mcp_remove_dropdown],
        [mcp_status_md]
    ).then(
        _refresh_mcp_dropdown, None, [mcp_remove_dropdown]
    )

    mcp_reconnect_btn.click(
        mcp_reconnect, None, [mcp_status_md]
    ).then(
        _refresh_mcp_dropdown, None, [mcp_remove_dropdown]
    )

    timer.tick(get_status, None, status_box)
    demo.load(get_status, None, status_box)
    demo.load(_refresh_mcp_dropdown, None, [mcp_remove_dropdown])
    demo.load(lambda: _build_mcp_status_text(), None, [mcp_status_md])

if __name__ == "__main__":
    demo.queue().launch(
        css=CUSTOM_CSS,
        theme=gr.themes.Monochrome(),
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        allowed_paths=[os.path.abspath("assets")]
    )
