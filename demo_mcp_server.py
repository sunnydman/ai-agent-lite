"""
极简 MCP 服务器 — 用于测试 AI Agent Lite 的 MCP 功能。
不依赖任何第三方库，纯 Python 标准库实现。

提供的工具:
  - get_weather: 模拟查天气
  - count_files: 统计目录中的文件数
  - current_time: 返回当前时间
"""

import sys
import json
import os
import datetime
from pathlib import Path


def handle_initialize(params):
    """响应 initialize 请求。"""
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": "demo-mcp-server",
            "version": "1.0.0"
        }
    }


def handle_tools_list(params):
    """返回可用工具列表。"""
    return {
        "tools": [
            {
                "name": "get_weather",
                "description": "查询指定城市的天气信息（模拟数据）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如 北京、上海、东京"
                        }
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "count_files",
                "description": "统计指定目录下的文件数量",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "要统计的目录路径"
                        }
                    },
                    "required": ["directory"]
                }
            },
            {
                "name": "current_time",
                "description": "获取当前的日期和时间",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }


def handle_tools_call(params):
    """处理工具调用。"""
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    if tool_name == "get_weather":
        city = arguments.get("city", "未知")
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "city": city,
                        "temperature": f"{hash(city) % 25 + 5}°C",
                        "condition": ["晴", "多云", "小雨", "阴"][hash(city) % 4],
                        "humidity": f"{hash(city) % 40 + 40}%",
                        "wind": f"{hash(city) % 5 + 1}级",
                        "note": "（本数据为 MCP 服务器模拟生成）"
                    }, ensure_ascii=False, indent=2)
                }
            ]
        }

    elif tool_name == "count_files":
        directory = arguments.get("directory", ".")
        try:
            path = Path(directory).expanduser().resolve()
            if not path.exists():
                text = f"目录不存在: {directory}"
            elif not path.is_dir():
                text = f"路径不是目录: {directory}"
            else:
                files = [f for f in path.iterdir() if f.is_file()]
                dirs = [d for d in path.iterdir() if d.is_dir()]
                text = json.dumps({
                    "directory": str(path),
                    "file_count": len(files),
                    "dir_count": len(dirs),
                    "files": [f.name for f in files[:20]],
                    "note": "（超过 20 个文件时仅显示前 20 个）"
                }, ensure_ascii=False, indent=2)
        except Exception as e:
            text = f"统计失败: {str(e)}"

        return {"content": [{"type": "text", "text": text}]}

    elif tool_name == "current_time":
        now = datetime.datetime.now()
        text = json.dumps({
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
            "timestamp": int(now.timestamp()),
        }, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": text}]}

    else:
        return {
            "content": [{"type": "text", "text": f"未知工具: {tool_name}"}],
            "isError": True
        }


def process_message(msg: dict) -> dict:
    """处理单条 JSON-RPC 消息。"""
    method = msg.get("method", "")
    req_id = msg.get("id")
    params = msg.get("params", {})

    try:
        if method == "initialize":
            result = handle_initialize(params)
        elif method == "tools/list":
            result = handle_tools_list(params)
        elif method == "tools/call":
            result = handle_tools_call(params)
        elif method == "notifications/initialized":
            return None  # 通知类消息不需要回复
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"未知方法: {method}"}
            }

        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32603, "message": str(e)}
        }


def main():
    """主循环：用 os.write 直写 fd，彻底绕过一切 Python 缓冲层。"""
    sys.stderr.write(f"[MCP Demo Server] Started PID={os.getpid()}\n")
    sys.stderr.write(f"[MCP Demo Server] Tools: get_weather, count_files, current_time\n")
    sys.stderr.flush()

    # 用底层 fd，不走 Python 文件对象
    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # 先写一个 startup 确认（证明 stdout 能写）
    os.write(stdout_fd, b'')  # no-op to test fd validity

    buf = b""
    msg_count = 0
    while True:
        try:
            chunk = os.read(stdin_fd, 4096)
        except OSError:
            break
        if not chunk:
            break

        buf += chunk
        while b"\n" in buf:
            nl = buf.index(b"\n")
            line_bytes = buf[:nl]
            buf = buf[nl + 1:]

            try:
                line = line_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                os.write(stderr_fd, f"[MCP] JSON err: {line_bytes[:80]}\n".encode())
                continue

            msg_count += 1
            method = msg.get("method", "?")
            os.write(stderr_fd, f"[MCP] RECV #{msg_count}: {method}\n".encode())

            response = process_message(msg)
            if response is not None:
                out = (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")
                os.write(stdout_fd, out)
                os.write(stderr_fd, f"[MCP] SENT #{msg_count} ({len(out)} bytes)\n".encode())


if __name__ == "__main__":
    main()
