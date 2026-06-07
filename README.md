# AI Agent Lite 大作业项目

这是一个面向《人工智能基础》大作业选题一的轻量版 AI Agent 框架。系统通过 Orchestrator、Planner、Executor 三层 Agent 协作，实现自然语言输入、工具选择、工具调用、观察结果回传和最终回答。

## 功能概览

- ReAct 循环：支持 `Thought -> Action -> Observation -> Final Answer` 多轮推理。
- 多工具系统：内置计算器、维基百科搜索、文件读取、文件写入、日期时间工具。
- MCP 扩展：支持从 `mcp_servers.json` 连接演示 MCP Server，并按 `mcp__服务器名__工具名` 注册远程工具。
- Shell Agent：提供只读命令演示模式，仅允许目录查看、文件查看、查找等查询类命令。
- Web 界面：基于 Gradio，支持多 Agent 节点会话和执行轨迹展示。
- TUI 界面：提供 `python tui.py` 命令行交互入口，支持 MCP 状态查看和 Shell 演示。
- 记忆系统：使用 SQLite FTS5 保存会话，并把相关长期记忆注入上下文。
- 安全边界：文件读写限制在工作目录内，避免路径穿越写入外部文件。

## 项目结构

```text
.
├── AGENTS.md          # Agent 行为准则
├── agents.py          # Orchestrator / Planner / Executor
├── llm_client.py      # OpenAI 兼容 API 客户端
├── memory.py          # SQLite FTS5 记忆管理
├── mcp_client.py      # MCP 客户端与远程工具注册
├── demo_mcp_server.py # 课程演示用 MCP Server
├── shell_agent.py     # 只读 Shell Agent 演示
├── tui.py             # Rich TUI 命令行界面
├── mcp_servers.json   # MCP Server 配置
├── tools.py           # 工具定义与注册表
├── web_ui.py          # Gradio Web 界面
├── requirements.txt   # 运行依赖
└── tests/             # 核心行为测试
```

## 安装依赖

建议使用 Python 3.10 或以上版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 启动方式

### Web 界面

```bash
python web_ui.py
```

如果提示 7860 端口被占用，说明本机已有程序正在使用默认端口。可以关闭占用该端口的程序，或在 `web_ui.py` 底部把 `server_port=7860` 改成其他端口，例如 `7861`，然后重新运行。

浏览器会打开本地页面。展开“核心配置”，填写 API Endpoint、API Key 和模型名称，然后点击“初始化节点”。

### TUI 界面

```bash
python tui.py
```

TUI 常用命令：

- `/mcp`：查看 MCP 服务器和远程工具状态。
- `/mcp-reload`：重新加载 `mcp_servers.json` 并刷新 MCP 工具。
- `/mcp-add <name> <command> [args...]`：添加 MCP Server，参数路径带空格时可使用引号。
- `/shell <自然语言>`：调用只读 Shell Agent，例如 `/shell 列出当前目录`。
- `/shell-rules`：查看 Shell Agent 的安全边界。

常用 API Endpoint：

- DeepSeek: `https://api.deepseek.com/v1`，模型 `deepseek-chat`
- Qwen: `https://dashscope.aliyuncs.com/compatible-mode/v1`，模型 `qwen-turbo`
- Kimi: `https://api.moonshot.cn/v1`，模型 `moonshot-v1-8k`
- 本地 Ollama: `http://localhost:11434/v1`，模型示例 `qwen2:7b`

## 推荐演示任务

```text
查一下爱因斯坦的出生年份和去世年份，然后算一下他活了多少岁
```

预期执行链路：

1. Orchestrator 接收任务并注入记忆。
2. Planner 判断任务需要外部工具。
3. Executor 先调用 `wiki_search` 查询人物信息。
4. Executor 再调用 `calculator` 计算 `1955 - 1879`。
5. Executor 输出最终答案。

也可以测试：

```text
计算 (2^10 + 3^5) * 7
当前日期时间是多少
把“hello agent”写入 notes/demo.txt，然后读取这个文件
```

## MCP 与 Shell Agent 说明

`mcp_servers.json` 默认启用 `demo` 服务器，提供 `get_weather`、`count_files`、`current_time` 三个演示工具。远程工具注册名采用 `mcp__服务器名__工具名`，例如：

```text
mcp__demo__current_time
```

Shell Agent 只作为课程演示功能，最终执行前会经过本地 allowlist 校验。允许命令包括 `dir`、`type`、`find`、`findstr`、`where` 等只读查询；拒绝写入、删除、安装、联网脚本执行、管道、重定向、命令连接符、解释器嵌套和越界路径。

## 与大作业要求对照

| 大作业要求 | 当前实现 |
|---|---|
| 至少 3 个工具 | 已实现 5 个内置工具，并可通过 MCP 扩展远程工具 |
| 自主选择合适工具 | Planner 判断是否需要工具，Executor 根据 ReAct 输出调用工具 |
| 多步调用任务 | 支持多轮工具调用，适合演示“搜索 + 计算” |
| 对话历史管理 | 保存最近短期记忆，并检索长期记忆注入上下文 |
| 不使用 LangChain/LlamaIndex | 未使用高层 Agent 框架 |
| Web 界面加分 | 已实现 Gradio Web UI |
| 扩展能力展示 | 新增 MCP 客户端、只读 Shell Agent 和 TUI |
| 长期记忆加分 | SQLite FTS5 持久化记忆 |
| 多 Agent 协作加分 | Orchestrator / Planner / Executor 三层结构 |

## 测试

项目使用 Python 标准库 `unittest`，不额外依赖测试框架。

```bash
python -m unittest discover -s tests -v
```

测试覆盖：

- 工具描述包含参数 schema。
- 文件写入拒绝越界路径。
- Orchestrator 注入长期记忆检索结果。
- Executor 可解析 JSON 格式工具调用。
- MCP demo 连接、工具发现、工具调用和重连复用。
- Shell Agent 只读边界、危险命令拒绝和超时终止。
- TUI/Web 管理入口的基础合同。

## 注意事项

- `memory.db` 是运行时生成文件，不应打进最终提交包。
- `__pycache__` 是 Python 缓存目录，不应提交。
- 文件读写工具默认把当前工作目录作为安全边界；也可以用环境变量 `AGENT_WORKSPACE` 指定允许读写的根目录。
- MCP 的第三方 Server 可能需要额外运行环境，例如 Node.js；默认 demo server 不依赖 Node.js。
