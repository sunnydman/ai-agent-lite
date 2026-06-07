"""
MCP (Model Context Protocol) Client — 让 AI Agent 连接外部 MCP 服务器，
自动发现并调用远程工具。

支持两种传输方式:
  - stdio: 启动本地子进程，通过标准输入/输出通信
  - http:  通过 HTTP + SSE 连接远程 MCP 服务器（TODO）

协议版本: 2024-11-05
"""

import json
import subprocess
import threading
import time
import uuid
import os
import sys
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path


# ── 数据结构 ───────────────────────────────────────────

@dataclass
class MCPServerConfig:
    """描述一个 MCP 服务器的连接方式。"""
    name: str
    command: str                       # 可执行文件路径，如 "npx" / "python"
    args: List[str] = field(default_factory=list)  # 命令行参数
    env: Dict[str, str] = field(default_factory=dict)  # 额外的环境变量
    enabled: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "enabled": self.enabled,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPServerConfig":
        return cls(
            name=d.get("name", ""),
            command=d.get("command", ""),
            args=d.get("args", []),
            env=d.get("env", {}),
            enabled=d.get("enabled", True),
            description=d.get("description", ""),
        )


@dataclass
class MCPToolSpec:
    """MCP 工具的描述信息，与内部 BaseTool.spec() 格式兼容。"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    server_name: str = ""              # 所属 MCP 服务器
    server_config: Optional[MCPServerConfig] = None

    def to_internal_spec(self) -> Dict[str, Any]:
        return {
            "name": f"mcp__{self.server_name}__{self.name}" if self.server_name else self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# ── JSON-RPC 消息构造 ──────────────────────────────────

def _rpc_request(method: str, params: Optional[Dict] = None, req_id: Any = None) -> Dict:
    if req_id is None:
        req_id = str(uuid.uuid4())
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}


def _rpc_notification(method: str, params: Optional[Dict] = None) -> Dict:
    return {"jsonrpc": "2.0", "method": method, "params": params or {}}


# ── Stdio 传输层 ───────────────────────────────────────

class StdioTransport:
    """通过子进程的 stdin/stdout 与 MCP 服务器通信。

    使用二进制管道 + 轮询读取，避免 Windows 上 text-mode 管道的
    readline() 阻塞/死锁问题。
    """

    def __init__(self, command: str, args: List[str], env: Dict[str, str],
                 cwd: Optional[str] = None, timeout: float = 30.0):
        self.command = command
        self.args = args
        self.env = env
        self.cwd = cwd
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._pending: Dict[Any, threading.Event] = {}
        self._results: Dict[Any, Dict] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._server_name = ""
        self._stdout_buf = b""

    def start(self, server_name: str = "") -> bool:
        self._server_name = server_name
        merged_env = {**os.environ, **self.env}
        merged_env.setdefault("PYTHONUNBUFFERED", "1")

        command = self.command
        args = list(self.args)

        if command in ("python", "python3"):
            command = sys.executable
            if "-u" not in args:
                args.insert(0, "-u")

        # 解析脚本相对路径
        config_dir = os.path.dirname(os.path.abspath(
            os.environ.get("MCP_CONFIG_PATH",
                           os.path.join(os.path.dirname(__file__), "mcp_servers.json"))
        ))
        resolved_args = []
        for a in args:
            if not os.path.isabs(a) and a.endswith(".py"):
                candidate = os.path.join(config_dir, a)
                if os.path.exists(candidate):
                    resolved_args.append(candidate)
                    continue
            resolved_args.append(a)

        try:
            # 🔧 用二进制模式打开管道，完全避免 text-mode 缓冲/编码问题
            self._process = subprocess.Popen(
                [command] + resolved_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                cwd=self.cwd or config_dir,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"MCP 服务器 [{server_name}] 启动失败: 找不到可执行文件 '{command}'。"
            )
        except Exception as e:
            raise RuntimeError(f"MCP 服务器 [{server_name}] 启动失败: {str(e)}")

        self._shutdown = False
        self._stdout_buf = b""
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()
        return True

    def send(self, message: Dict, timeout: Optional[float] = None) -> Dict:
        if not self._process or self._process.poll() is not None:
            raise RuntimeError(f"MCP 服务器 [{self._server_name}] 已断开")

        req_id = message.get("id")
        if req_id is None:
            with self._lock:
                self._write_line(json.dumps(message, ensure_ascii=False))
            return {}

        event = threading.Event()
        with self._lock:
            self._pending[req_id] = event
            self._write_line(json.dumps(message, ensure_ascii=False))

        effective_timeout = timeout or self.timeout
        if not event.wait(timeout=effective_timeout):
            with self._lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(
                f"MCP 服务器 [{self._server_name}] 响应超时 ({effective_timeout}s): {message.get('method')}"
            )

        with self._lock:
            result = self._results.pop(req_id, None)
        if result is None:
            raise RuntimeError(f"MCP [{self._server_name}] 未收到响应: {message.get('method')}")
        if "error" in result:
            err = result["error"]
            raise RuntimeError(
                f"MCP [{self._server_name}] 返回错误: {err.get('message', str(err))}"
            )
        return result.get("result", {})

    def stop(self):
        self._shutdown = True
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        if self._process:
            try:
                self._process.stdin.close()
                self._process.stdout.close()
                self._process.stderr.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _write_line(self, line: str):
        if self._process and self._process.stdin and not self._process.stdin.closed:
            try:
                data = (line + "\n").encode("utf-8")
                self._process.stdin.write(data)
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def _read_loop(self):
        """从 stdout 逐字节读取，手动拆分成 JSON 行。"""
        while not self._shutdown and self._process and self._process.stdout:
            try:
                # 用 os.read 分段读（比 readline 更可控）
                chunk = os.read(self._process.stdout.fileno(), 4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break  # EOF

            self._stdout_buf += chunk

            # 从缓冲区中提取完整的行
            while True:
                nl = self._stdout_buf.find(b"\n")
                if nl == -1:
                    break
                line_bytes = self._stdout_buf[:nl]
                self._stdout_buf = self._stdout_buf[nl + 1:]

                try:
                    line = line_bytes.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                req_id = msg.get("id")
                if req_id is not None:
                    with self._lock:
                        self._results[req_id] = msg
                        event = self._pending.pop(req_id, None)
                    if event:
                        event.set()

    def _stderr_reader(self):
        while not self._shutdown and self._process and self._process.stderr:
            try:
                chunk = os.read(self._process.stderr.fileno(), 4096)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            # 转发到控制台便于调试
            try:
                sys.stderr.write(chunk.decode("utf-8", errors="replace"))
                sys.stderr.flush()
            except Exception:
                pass


# ── MCP 客户端（管理一个服务器连接）────────────────────

class MCPClient:
    """管理与单个 MCP 服务器的连接，提供工具发现与调用。"""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.transport: Optional[StdioTransport] = None
        self._tools: Dict[str, MCPToolSpec] = {}
        self._connected = False
        self._server_capabilities: Dict = {}

    # ── 连接管理 ───────────────────────────────────

    def connect(self) -> bool:
        if self._connected:
            return True

        self.transport = StdioTransport(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
        )
        self.transport.start(self.config.name)

        try:
            init_result = self.transport.send(_rpc_request(
                "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ai-agent-lite", "version": "1.1.0"},
                }
            ))
            self._server_capabilities = init_result.get("capabilities", {})

            # 发送 initialized 通知
            self.transport.send(_rpc_notification("notifications/initialized"), timeout=5)

            self._connected = True
        except Exception:
            self.transport.stop()
            self.transport = None
            raise

        return True

    def disconnect(self):
        if self.transport:
            try:
                self.transport.stop()
            except Exception:
                pass
            self.transport = None
        self._connected = False
        self._tools.clear()
        self._server_capabilities.clear()

    @property
    def is_connected(self) -> bool:
        return self._connected and (self.transport and self.transport.is_alive)

    # ── 工具发现 ───────────────────────────────────

    def discover_tools(self) -> List[MCPToolSpec]:
        if not self._connected:
            raise RuntimeError(f"MCP 服务器 [{self.config.name}] 未连接")

        result = self.transport.send(_rpc_request("tools/list", {}))
        raw_tools = result.get("tools", [])

        self._tools.clear()
        specs = []
        for raw in raw_tools:
            spec = MCPToolSpec(
                name=raw.get("name", ""),
                description=raw.get("description", ""),
                parameters=raw.get("inputSchema", {}),
                server_name=self.config.name,
                server_config=self.config,
            )
            self._tools[spec.name] = spec
            specs.append(spec)

        return specs

    def get_tools(self) -> Dict[str, MCPToolSpec]:
        return dict(self._tools)

    # ── 工具调用 ───────────────────────────────────

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self._connected:
            raise RuntimeError(f"MCP 服务器 [{self.config.name}] 未连接")
        if tool_name not in self._tools:
            raise ValueError(
                f"工具 '{tool_name}' 不在 MCP 服务器 [{self.config.name}] 中。"
                f"可用工具: {list(self._tools.keys())}"
            )

        result = self.transport.send(_rpc_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        ))

        # 提取文本内容
        content = result.get("content", [])
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif item.get("type") == "resource":
                        text_parts.append(f"[Resource: {item.get('resource', {})}]")
                    else:
                        text_parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    text_parts.append(str(item))
            return "\n".join(text_parts) if text_parts else json.dumps(result, ensure_ascii=False)
        return str(content)

    # ── 资源（可选）───────────────────────────────────

    def list_resources(self) -> List[Dict]:
        if not self._connected:
            return []
        try:
            result = self.transport.send(_rpc_request("resources/list", {}))
            return result.get("resources", [])
        except Exception:
            return []

    def read_resource(self, uri: str) -> str:
        if not self._connected:
            raise RuntimeError(f"MCP 服务器 [{self.config.name}] 未连接")
        result = self.transport.send(_rpc_request(
            "resources/read", {"uri": uri}
        ))
        contents = result.get("contents", [])
        text_parts = []
        for item in contents:
            if isinstance(item, dict):
                text_parts.append(item.get("text", str(item)))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)


# ── MCP 管理器（管理多个 MCP 服务器）───────────────────

class MCPManager:
    """管理多个 MCP 服务器连接，聚合所有远程工具。"""

    def __init__(self, configs: Optional[List[MCPServerConfig]] = None):
        self._clients: Dict[str, MCPClient] = {}
        self._configs: Dict[str, MCPServerConfig] = {}
        if configs:
            for cfg in configs:
                self._configs[cfg.name] = cfg

    # ── 配置管理 ───────────────────────────────────

    def add_server(self, config: MCPServerConfig):
        self._configs[config.name] = config
        # 如果已有同名的旧连接，断开
        if config.name in self._clients:
            self._clients[config.name].disconnect()
            del self._clients[config.name]

    def remove_server(self, name: str):
        self._configs.pop(name, None)
        if name in self._clients:
            self._clients[name].disconnect()
            del self._clients[name]

    def get_configs(self) -> List[MCPServerConfig]:
        return list(self._configs.values())

    # ── 连接管理 ───────────────────────────────────

    def connect_all(self) -> Dict[str, bool]:
        results = {}
        for name, cfg in self._configs.items():
            if not cfg.enabled:
                results[name] = False
                continue
            try:
                existing = self._clients.get(name)
                if existing and existing.is_connected:
                    results[name] = True
                    continue
                if existing:
                    existing.disconnect()
                client = MCPClient(cfg)
                client.connect()
                self._clients[name] = client
                results[name] = True
            except Exception as e:
                results[name] = False
                print(f"[MCP] 连接 {name} 失败: {e}")
        return results

    def disconnect_all(self):
        for name, client in list(self._clients.items()):
            try:
                client.disconnect()
            except Exception:
                pass
        self._clients.clear()

    def get_server_status(self) -> Dict[str, Dict]:
        status = {}
        for name, cfg in self._configs.items():
            client = self._clients.get(name)
            status[name] = {
                "configured": True,
                "enabled": cfg.enabled,
                "connected": client.is_connected if client else False,
                "tool_count": len(client._tools) if client else 0,
                "description": cfg.description,
            }
        return status

    # ── 工具聚合 ───────────────────────────────────

    def discover_all_tools(self) -> List[MCPToolSpec]:
        all_tools = []
        for name, client in self._clients.items():
            if not client.is_connected:
                continue
            try:
                tools = client.discover_tools()
                all_tools.extend(tools)
            except Exception as e:
                print(f"[MCP] 发现工具失败 ({name}): {e}")
        return all_tools

    def find_tool(self, qualified_name: str) -> Optional[tuple]:
        """根据完整限定名 (如 mcp__filesystem__read_file) 找到对应的客户端和工具名。"""
        for client_name, client in self._clients.items():
            prefix = f"mcp__{client_name}__"
            if qualified_name.startswith(prefix):
                tool_name = qualified_name[len(prefix):]
                if tool_name in client._tools:
                    return client, tool_name
        return None

    def call_tool(self, qualified_name: str, arguments: Dict[str, Any]) -> str:
        found = self.find_tool(qualified_name)
        if not found:
            raise ValueError(f"未找到 MCP 工具: {qualified_name}")
        client, tool_name = found
        return client.call_tool(tool_name, arguments)

    def get_all_tool_specs_for_prompt(self) -> str:
        """生成所有 MCP 工具的描述文本，用于注入 system prompt。"""
        all_specs = []
        for client_name, client in self._clients.items():
            if not client.is_connected:
                continue
            for tool_name, tool in client._tools.items():
                qualified = f"mcp__{client_name}__{tool_name}"
                all_specs.append({
                    "name": qualified,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
        if not all_specs:
            return ""
        return json.dumps(all_specs, ensure_ascii=False, indent=2)

    # ── 持久化 ────────────────────────────────────

    def save_configs(self, path: str):
        data = {
            "servers": [cfg.to_dict() for cfg in self._configs.values()],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_configs(cls, path: str) -> "MCPManager":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        configs = [MCPServerConfig.from_dict(s) for s in data.get("servers", [])]
        return cls(configs)


# ── 便捷函数：从 JSON 配置文件创建 MCPManager ──────────

_MCP_CONFIG_PATH = Path(__file__).parent / "mcp_servers.json"


def get_mcp_manager(config_path: Optional[str] = None) -> MCPManager:
    """获取全局 MCP 管理器实例（懒加载）。"""
    path = config_path or str(_MCP_CONFIG_PATH)
    return MCPManager.load_configs(path)
