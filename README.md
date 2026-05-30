# AI Agent Lite 大作业项目

这是一个面向《人工智能基础》大作业选题一的轻量版 AI Agent 框架。系统通过 Orchestrator、Planner、Executor 三层 Agent 协作，实现自然语言输入、工具选择、工具调用、观察结果回传和最终回答。

## 功能概览

- ReAct 循环：支持 `Thought -> Action -> Observation -> Final Answer` 多轮推理。
- 多工具系统：内置计算器、维基百科搜索、文件读取、文件写入、日期时间工具。
- Web 界面：基于 Gradio，支持多 Agent 节点会话和执行轨迹展示。
- 记忆系统：使用 SQLite FTS5 保存会话，并把相关长期记忆注入上下文。
- 安全边界：文件读写限制在工作目录内，避免路径穿越写入外部文件。

## 项目结构

```text
.
├── AGENTS.md          # Agent 行为准则
├── agents.py          # Orchestrator / Planner / Executor
├── llm_client.py      # OpenAI 兼容 API 客户端
├── memory.py          # SQLite FTS5 记忆管理
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

```bash
python web_ui.py
```

如果提示 7860 端口被占用，说明本机已有程序正在使用默认端口。可以关闭占用该端口的程序，或在 `web_ui.py` 底部把 `server_port=7860` 改成其他端口，例如 `7861`，然后重新运行。

浏览器会打开本地页面。展开“核心配置”，填写 API Endpoint、API Key 和模型名称，然后点击“初始化节点”。

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

## 与大作业要求对照

| 大作业要求 | 当前实现 |
|---|---|
| 至少 3 个工具 | 已实现 5 个工具 |
| 自主选择合适工具 | Planner 判断是否需要工具，Executor 根据 ReAct 输出调用工具 |
| 多步调用任务 | 支持多轮工具调用，适合演示“搜索 + 计算” |
| 对话历史管理 | 保存最近短期记忆，并检索长期记忆注入上下文 |
| 不使用 LangChain/LlamaIndex | 未使用高层 Agent 框架 |
| Web 界面加分 | 已实现 Gradio Web UI |
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

## 注意事项

- `memory.db` 是运行时生成文件，不应打进最终提交包。
- `__pycache__` 是 Python 缓存目录，不应提交。
- 文件读写工具默认把当前工作目录作为安全边界；也可以用环境变量 `AGENT_WORKSPACE` 指定允许读写的根目录。
