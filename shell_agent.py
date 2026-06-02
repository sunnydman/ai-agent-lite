"""
Shell Agent — 自然语言转命令执行，双重安全机制。

组件:
  SafetyRuleEngine   — 本地规则引擎，拦截高风险命令
  ShellCommandExecutor — 子进程流式执行
  ShellAgent         — 自然语言→LLM结构化命令→安全校验→执行

安全检查流程（双重校验）:
  NL输入 → LLM生成{command, risk_level} → 本地规则引擎检查
    ├─ (LLM low + 本地 clean) → 直接执行
    ├─ (LLM risky / 本地拦截) → 要求用户确认
    └─ 用户确认后 → 流式执行
"""

import re
import subprocess
import platform
import os
import shlex
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Tuple

# ── 数据结构 ───────────────────────────────────────────

@dataclass
class SafetyVerdict:
    """本地规则引擎的检查结果。"""
    safe: bool
    reason: str                 # 拦截原因（safe 时为空）
    matched_rules: List[str] = field(default_factory=list)
    requires_confirmation: bool = False


@dataclass
class CommandResult:
    """命令执行结果。"""
    success: bool
    command: str
    stdout: str
    stderr: str
    exit_code: int
    risk_level: str = "low"
    was_confirmed: bool = False


# ── 危险命令规则（跨平台） ──────────────────────────────

# 格式: (正则模式, 风险等级, 描述)
_DANGER_RULES: List[Tuple[str, str, str]] = [
    # ── 不可逆删除 ──
    (r"\brm\s+-rf\b", "high", "递归强制删除 (rm -rf)"),
    (r"\brm\s+-r\b.*\b/", "high", "递归删除涉及根目录"),
    (r"\brmdir\b", "medium", "删除目录"),
    (r"\bdel\s+/[fFqQ].*\b(SYSTEM32|WINDOWS|system32)\b", "high", "删除系统目录"),
    (r"\bRemove-Item\s+-Recurse\s+-Force\b", "high", "PowerShell 递归强制删除"),

    # ── 权限提升 ──
    (r"\bsudo\b", "high", "提权操作 (sudo)"),
    (r"\bsu\s+-", "high", "切换用户 (su)"),
    (r"\bRunAs\b", "high", "Windows 提权 (RunAs)"),

    # ── 磁盘/文件系统破坏 ──
    (r"\bmkfs\.", "high", "格式化文件系统 (mkfs)"),
    (r"\bdd\s+if=", "high", "磁盘直接写入 (dd)"),
    (r"\bformat\s+[A-Za-z]:", "high", "格式化磁盘"),
    (r"\bdiskpart\b", "high", "磁盘分区工具"),
    (r"\bchmod\s+777", "medium", "开放所有权限 (chmod 777)"),
    (r"\bchmod\s+-R\s+777\b", "high", "递归开放所有权限"),
    (r"\bchown\s+-R\b.*\b/", "high", "递归变更根目录所有者"),
    (r"\bicacls\b.*/grant.*Everyone.*F", "high", "Windows 开放所有权限"),

    # ── 数据库破坏 ──
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", "high", "删除数据库/表 (DROP)"),
    (r"\bTRUNCATE\s+(TABLE\s+)?", "high", "清空表数据 (TRUNCATE)"),
    (r"\bDELETE\s+FROM\b", "medium", "删除数据库记录 (DELETE)"),

    # ── 系统修改 ──
    (r"\bshutdown\b", "medium", "关机/重启"),
    (r"\breboot\b", "medium", "重启"),
    (r"\bkill\s+-9\b", "medium", "强制杀死进程 (kill -9)"),
    (r"\bpkill\b", "medium", "批量杀死进程"),
    (r"\bsc\s+stop\b", "medium", "停止 Windows 服务"),
    (r"\bsc\s+delete\b", "high", "删除 Windows 服务"),

    # ── 网络风险 ──
    (r"\bcurl.*\|.*(sh|bash|python)\b", "high", "管道执行远程脚本"),
    (r"\bwget.*\|.*(sh|bash)\b", "high", "管道执行远程脚本"),
    (r"\biptables\s+-F\b", "high", "清空防火墙规则"),
    (r"\bufw\s+disable\b", "high", "禁用防火墙"),

    # ── 脚本/编码执行 ──
    (r"\beval\b", "medium", "动态代码执行 (eval)"),
    (r"\bexec\b", "medium", "进程替换执行 (exec)"),
    (r"\bInvoke-Expression\b", "high", "PowerShell 动态执行 (IEX)"),
    (r"\bStart-Process\b.*-Verb\s+RunAs", "high", "PowerShell 提权启动"),
]

# 允许的目录 (Windows)
_SAFE_PATHS_WIN = [
    r"C:\\Users\\", r"D:\\", r"E:\\",
    r"%USERPROFILE%", r"%HOME%", r"%TEMP%",
    r"\.\\", r"\.\.\\",
]

# 允许的目录 (Unix)
_SAFE_PATHS_UNIX = [
    "/home/", "/Users/", "/tmp/",
    "$HOME", "~/", "./", "../",
]


# ── 安全规则引擎 ───────────────────────────────────────

class SafetyRuleEngine:
    """本地规则引擎 — 纯模式匹配，不依赖 LLM，零延迟。"""

    def __init__(self):
        self._rules = _DANGER_RULES
        self._is_windows = platform.system() == "Windows"

    def check(self, command: str) -> SafetyVerdict:
        """检查命令是否安全。返回 SafetyVerdict。"""
        cmd_lower = command.lower()
        matched_rules = []
        highest_risk = "low"

        for pattern, risk_level, description in _DANGER_RULES:
            if re.search(pattern, cmd_lower, re.IGNORECASE):
                matched_rules.append(f"[{risk_level}] {description}: matched '{pattern}'")
                if risk_level == "high":
                    highest_risk = "high"
                elif risk_level == "medium" and highest_risk != "high":
                    highest_risk = "medium"

        if matched_rules and highest_risk == "high":
            return SafetyVerdict(
                safe=False,
                reason="检测到高风险命令模式",
                matched_rules=matched_rules,
                requires_confirmation=True,
            )
        elif matched_rules and highest_risk == "medium":
            return SafetyVerdict(
                safe=False,
                reason="检测到中等风险命令",
                matched_rules=matched_rules,
                requires_confirmation=True,
            )

        # 自定义安全目录检查（仅针对文件操作命令）
        if self._is_windows:
            self._check_path_traversal_windows(command, matched_rules)
        else:
            self._check_path_traversal_unix(command, matched_rules)

        if matched_rules:
            return SafetyVerdict(
                safe=False,
                reason="检测到路径相关风险",
                matched_rules=matched_rules,
                requires_confirmation=False,  # 中等风险给个提示但不强制确认
            )

        return SafetyVerdict(safe=True, reason="", matched_rules=[])

    def _check_path_traversal_unix(self, command: str, matched_rules: List[str]):
        for path_arg in re.findall(r'(?:^|\s)(/(?:[^/\s]+/)*[^/\s]+)', command):
            # 敏感系统路径之外还看是否纯相对路径
            sensitive = ["/etc/", "/var/", "/root/", "/bin/", "/sbin/", "/usr/", "/boot/", "/sys/", "/proc/"]
            for s in sensitive:
                if path_arg.startswith(s):
                    matched_rules.append(f"[medium] 操作系统目录: {path_arg}")
                    return

    def _check_path_traversal_windows(self, command: str, matched_rules: List[str]):
        sensitive = ["C:\\Windows", "C:\\Program Files", "C:\\ProgramData",
                     "SYSTEM32", "C:\\Boot", "C:\\Recovery"]
        cmd_upper = command.upper()
        for s in sensitive:
            if s.upper() in cmd_upper:
                matched_rules.append(f"[medium] 操作系统目录: {s}")
                return

    def list_rules(self) -> List[str]:
        """列出所有已知的危险模式。"""
        return [f"{risk_level:6s} | {desc}" for pattern, risk_level, desc in self._rules]


# ── 命令执行器 ─────────────────────────────────────────

class ShellCommandExecutor:
    """安全的子进程命令执行器，支持流式输出。"""

    SHELL_WIN = "cmd.exe"
    SHELL_UNIX = "/bin/bash"

    def __init__(self, timeout: int = 60, max_output_lines: int = 500, cwd: str = None):
        self.timeout = timeout
        self.max_output_lines = max_output_lines
        self.cwd = cwd or os.getcwd()  # 固定工作目录，避免子进程跑到别处

    def execute_stream(self, command: str) -> Iterator[str]:
        """流式执行命令，逐行 yield stdout（stderr 合并）。"""
        if platform.system() == "Windows":
            # 使用 Windows 原生编码 cp936（GBK），避免 chcp 65001 不生效导致乱码
            popen_args = {
                "args": command,
                "shell": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "cp936",
                "errors": "replace",
                "cwd": self.cwd,
                "creationflags": subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            }
        else:
            popen_args = {
                "args": ["/bin/bash", "-c", command],
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "cwd": self.cwd,
            }

        try:
            process = subprocess.Popen(**popen_args)
        except Exception as e:
            yield f"[ShellAgent] 启动失败: {str(e)}"
            return

        lines = 0
        try:
            for line in process.stdout:
                yield line.rstrip("\n")
                lines += 1
                if lines >= self.max_output_lines:
                    process.kill()
                    yield f"\n[ShellAgent] 输出超过 {self.max_output_lines} 行，已截断。"
                    break
        except Exception as e:
            yield f"[ShellAgent] 读取中断: {str(e)}"
        finally:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def execute(self, command: str) -> CommandResult:
        """非流式执行，返回完整 CommandResult。"""

        if platform.system() == "Windows":
            popen_args = {
                "args": command,
                "shell": True,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "cp936",
                "errors": "replace",
                "cwd": self.cwd,
                "creationflags": subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            }
        else:
            popen_args = {
                "args": ["/bin/bash", "-c", command],
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "cwd": self.cwd,
            }

        try:
            process = subprocess.Popen(**popen_args)
            out, err = process.communicate(timeout=self.timeout)
            return CommandResult(
                success=process.returncode == 0,
                command=command,
                stdout=out.strip() if out else "",
                stderr=err.strip() if err else "",
                exit_code=process.returncode,
            )
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except Exception:
                pass
            return CommandResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"命令超时 ({self.timeout}s)",
                exit_code=-1,
            )
        except FileNotFoundError:
            return CommandResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"找不到命令: {command.split()[0] if command.split() else command}",
                exit_code=-1,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                command=command,
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )


# ── Shell Agent ────────────────────────────────────────

class ShellAgent:
    """
    自然语言 → 命令 的 Shell Agent。

    流程:
      用户说 "帮我看看当前目录" →
      LLM 生成 {"intent":"run_command", "command":"dir", "risk_level":"low"} →
      本地规则引擎校验 →
      如果低风险 → 直接执行 |
      如果高风险 → 要求用户确认 →
      流式输出执行结果
    """

    def __init__(self, llm_client, timeout: int = 60, cwd: str = None):
        self.llm = llm_client
        self.safety = SafetyRuleEngine()
        self.executor = ShellCommandExecutor(timeout=timeout, cwd=cwd)
        self._pending_confirmation: Optional[Dict] = None

    # ── LLM 结构化 ────────────────────────────────

    def _build_nl2cmd_prompt(self, user_input: str) -> str:
        """构建自然语言转命令的 prompt。"""
        return f"""你是一个安全的 Shell 命令生成器。根据用户的自然语言请求，生成结构化的 JSON 命令。

当前系统: {platform.system()}
系统版本: {platform.version()}
当前目录: {os.getcwd()}

【规则】
1. 优先使用系统内置命令（Windows: dir/cd/copy/move/mkdir/find/findstr/type; Unix: ls/cd/cp/mv/mkdir/find/grep/cat）
2. 只输出一个 JSON 对象，不要有多余内容
3. risk_level 评估:
   - "low": 只读操作（列文件、查看内容、查找、统计）
   - "medium": 创建/移动文件，修改配置
   - "high": 删除、权限变更、系统修改、可能不可逆的操作
4. intent:
   - "run_command": 可以安全执行
   - "ask_clarification": 不确定用户意图，需要追问
   - "refuse": 明显危险、违反安全规则，拒绝执行

用户请求: {user_input}

请输出 JSON:
{{"intent": "run_command|ask_clarification|refuse", "command": "...", "reason": "原因说明", "risk_level": "low|medium|high"}}"""

    def _parse_llm_command(self, raw: str) -> dict:
        """从 LLM 回复中提取 JSON 命令。"""
        # 尝试多种方式提取 JSON
        for pattern in [
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"(\{[^{}]*\"intent\"[^{}]*\"command\"[^{}]*\})",
            r"(\{.*\"intent\".*\})",
        ]:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # 最后尝试整段文本
        try:
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                return json.loads(raw[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

        return {}

    def analyze(self, user_input: str) -> dict:
        """分析自然语言请求，生成结构化命令。"""
        prompt = self._build_nl2cmd_prompt(user_input)
        raw = self.llm.chat([{"role": "user", "content": prompt}], stream=False)
        result = self._parse_llm_command(raw)
        if not result:
            # LLM 解析失败 → 安全兜底
            return {
                "intent": "ask_clarification",
                "command": "",
                "reason": "无法解析用户意图，请换一种方式描述。",
                "risk_level": "low",
            }
        return result

    # ── 安全校验 ──────────────────────────────────

    def check_safety(self, command_info: dict) -> Tuple[SafetyVerdict, bool]:
        """双重安全校验：LLM risk_level + 本地规则引擎。

        返回 (本地裁决, 是否需要用户确认)。
        """
        command = command_info.get("command", "")
        llm_risk = command_info.get("risk_level", "low")

        # 第一层: LLM 判断
        if command_info.get("intent") == "refuse":
            return (
                SafetyVerdict(
                    safe=False,
                    reason=f"LLM 拒绝执行: {command_info.get('reason', '')}",
                    matched_rules=[],
                    requires_confirmation=False,
                ),
                False,  # 直接拒绝，不需要确认
            )

        if command_info.get("intent") == "ask_clarification":
            return (
                SafetyVerdict(
                    safe=False,
                    reason=f"LLM 需要澄清: {command_info.get('reason', '')}",
                    matched_rules=[],
                    requires_confirmation=False,
                ),
                False,
            )

        if not command.strip():
            return (
                SafetyVerdict(
                    safe=False,
                    reason="命令为空",
                    matched_rules=[],
                    requires_confirmation=False,
                ),
                False,
            )

        # 第二层: 本地规则引擎
        local_verdict = self.safety.check(command)

        # 合并判断
        needs_confirmation = False
        if llm_risk == "high":
            needs_confirmation = True
        if not local_verdict.safe and local_verdict.requires_confirmation:
            needs_confirmation = True
        if llm_risk == "medium" and not local_verdict.safe:
            needs_confirmation = True

        return local_verdict, needs_confirmation

    # ── 执行 ──────────────────────────────────────

    def execute_streaming(self, command: str) -> Iterator[str]:
        """流式执行命令。"""
        yield from self.executor.execute_stream(command)

    def execute(self, command: str) -> CommandResult:
        """非流式执行。"""
        result = self.executor.execute(command)
        result.was_confirmed = True
        return result

    # ── 完整流程 ──────────────────────────────────

    def run_pipeline(
        self, user_input: str
    ) -> Tuple[dict, SafetyVerdict, bool, Optional[CommandResult]]:
        """
        完整管道:
          NL → LLM结构化 → 安全校验 → 返回结果
        返回 (命令信息, 安全裁决, 是否需要确认, 执行结果[如果已执行])

        TUI/WebUI 调用此方法：
          1. 看 needs_confirmation → 如果是 True，弹确认框
          2. 用户确认后 → 调用 self.execute()
        """
        cmd_info = self.analyze(user_input)
        safety_verdict, needs_confirmation = self.check_safety(cmd_info)

        if cmd_info["intent"] in ("refuse", "ask_clarification") or not cmd_info["command"].strip():
            return cmd_info, safety_verdict, False, None

        if needs_confirmation:
            self._pending_confirmation = cmd_info
            return cmd_info, safety_verdict, True, None

        # 安全 → 直接执行
        result = self.execute(cmd_info["command"])
        result.risk_level = cmd_info.get("risk_level", "low")
        return cmd_info, safety_verdict, False, result

    def execute_pending(self) -> Optional[CommandResult]:
        """执行挂起待确认的命令。"""
        if self._pending_confirmation is None:
            return None
        cmd_info = self._pending_confirmation
        self._pending_confirmation = None
        result = self.execute(cmd_info["command"])
        result.risk_level = cmd_info.get("risk_level", "low")
        result.was_confirmed = True
        return result

    def get_pending_command(self) -> Optional[str]:
        """返回待确认的命令文本。"""
        if self._pending_confirmation:
            return self._pending_confirmation["command"]
        return None


# ── 便捷工厂 ───────────────────────────────────────────

def create_shell_agent(llm_client, cwd: str = None) -> ShellAgent:
    return ShellAgent(llm_client, cwd=cwd)
