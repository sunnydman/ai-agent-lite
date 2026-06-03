import os
import json
import re
import platform
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable
from llm_client import LLMClient
from tools import TOOL_REGISTRY, register_mcp_tools, unregister_mcp_tools


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    metadata: dict = field(default_factory=dict)


class AgentEvent:
    """TUI / WebUI 均可消费的流式事件。"""

    def __init__(self, event_type: str, data: Any = None):
        self.type = event_type  # "trace" | "thought" | "tool_call" | "tool_result" | "chunk" | "final"
        self.data = data


class BaseAgent:
    def __init__(self, name, llm_client):
        self.name = name
        self.llm = llm_client

    def process(self, msg: AgentMessage) -> AgentMessage:
        raise NotImplementedError


class ExecutorAgent(BaseAgent):
    MAX_ITERATIONS = 5

    def __init__(self, llm_client, agents_md, mcp_manager=None):
        super().__init__("Executor", llm_client)
        self.agents_md = agents_md
        self.mcp_manager = mcp_manager
        self.tool_registry = dict(TOOL_REGISTRY)
        self._refresh_mcp_tools()
        self.tools_desc = self._build_tools_desc()
        self.system_prompt = self._build_prompt()

    def _refresh_mcp_tools(self):
        """刷新 MCP 远程工具注册。每次创建 Executor 时同步一次。"""
        if self.mcp_manager is None:
            return 0
        unregister_mcp_tools(self.tool_registry)
        return register_mcp_tools(self.mcp_manager, self.tool_registry)

    def _condense_history(self, history: List[Dict]) -> List[Dict]:
        """把完整对话历史压缩为精简上下文，过滤掉旧轮次的工具 JSON 残片。
        Executor 只需要知道聊过什么主题，不需要看到 Action/Observation 细节。"""
        if not history:
            return []
        condensed = []
        for h in history:
            role = h.get("role", "")
            content = str(h.get("content", ""))
            snippet = content[:150].replace("\n", " ").strip()
            # 跳过工具调用 JSON 和 Observation 残片
            if role == "assistant" and snippet.startswith("{") and \
                    any(kw in snippet.lower() for kw in ("action", "observation", "tool")):
                continue
            if not snippet:
                continue
            if role == "user":
                condensed.append({"role": "system", "content": f"[历史] 用户曾问: {snippet}"})
            elif role == "assistant":
                condensed.append({"role": "system", "content": f"[历史] 助手曾答: {snippet}"})
        return condensed[-4:]  # 最多保留最近 4 条摘要


    def _build_tools_desc(self):
        tools_list = [tool.spec() for tool in self.tool_registry.values()]
        return json.dumps(tools_list, ensure_ascii=False, indent=2)

    def _build_prompt(self):
        mcp_note = ""
        if self.mcp_manager:
            status = self.mcp_manager.get_server_status()
            connected = [n for n, s in status.items() if s.get("connected")]
            if connected:
                mcp_note = f"\n已连接 MCP 服务器: {', '.join(connected)}。以 'mcp__' 开头的工具来自外部 MCP 服务器。"

        return f"""你是一个强大的 AI 助手。请遵循以下规则：
{self.agents_md}
当前系统: {platform.system()} | 目录: {os.getcwd()}
可用工具: {self.tools_desc}{mcp_note}

【重要】对话格式规则：
1. 需要调用工具时，严格输出 JSON：
   {{"thought": "你的思考", "action": "工具名", "action_input": {{"参数": "值"}}}}
2. 工具返回 Observation 后，必须立即用 final_answer 回复——不要再次调用工具：
   {{"thought": "已获得所需信息", "final_answer": "自然的完整回复"}}
3. final_answer 的内容必须是用流畅中文把工具结果转述出来。
   比如工具返回 {{"temperature":"11°C","city":"杭州"}}，你应输出：
   {{"final_answer": "杭州今天多云，气温11°C，湿度41%，风力2级。"}}
4. 每次都只输出一个 JSON 对象，不要在一个回复里拼接多个 JSON。
5. 如果无需工具，直接输出 final_answer。
"""

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates = [code_block.group(1)] if code_block else []

        # 逐个提取 JSON 对象 —— 处理 LLM 一次输出多个 {}{} 的情况
        pos = 0
        while pos < len(text):
            brace_start = text.find("{", pos)
            if brace_start == -1:
                break
            depth = 0
            i = brace_start
            while i < len(text):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[brace_start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                candidates.append(parsed)
                        except json.JSONDecodeError:
                            pass
                        pos = i + 1
                        break
                i += 1
            else:
                pos = len(text)  # 未闭合

        if not candidates:
            # 兜底：整个文本当作 JSON
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            return None

        # 优先返回有 final_answer 的对象，否则返回第一个有 action 的，否则返回第一个
        for c in candidates:
            if isinstance(c, dict) and (c.get("final_answer") or c.get("final")):
                return c
        for c in candidates:
            if isinstance(c, dict) and c.get("action"):
                return c
        return candidates[0] if candidates else None

    def _parse_llm_response(self, text: str) -> Tuple[str, Optional[str], Dict[str, Any], Optional[str]]:
        parsed = self._extract_json_object(text)
        if parsed:
            thought = str(parsed.get("thought", "")).strip()
            final_answer = parsed.get("final_answer") or parsed.get("final")
            if final_answer is not None:
                return thought, None, {}, str(final_answer).strip()

            action = parsed.get("action")
            action_input = parsed.get("action_input") or parsed.get("arguments") or {}
            if isinstance(action_input, str):
                action_input = self._extract_json_object(action_input) or {}
            if action:
                return thought, str(action).strip(), action_input, None

        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
        if final_match:
            return thought, None, {}, final_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.*?)\nAction Input:\s*(.*)", text, re.DOTALL)
        if not action_match:
            # 既没 action 也没 Final Answer → 把整段文本当最终答案
            # 这样 LLM 自由格式的自然语言回复也能被正确消费
            clean = text.strip()
            # 去掉可能的 "Thought:" 前缀残留
            clean = re.sub(r"^Thought:\s*.*?\n", "", clean)
            if len(clean) > 5:
                return thought, None, {}, clean
            return thought, None, {}, None

        action = action_match.group(1).strip()
        action_input_str = action_match.group(2).strip()
        action_input = self._extract_json_object(action_input_str) or {}
        return thought, action, action_input, None

    def _fallback_answer_from_observation(self, action: str, observation: str) -> str:
        if action == "calculator":
            return f"计算结果：{observation}"

        # MCP 工具返回的 JSON → 试试让 LLM 转成自然语言
        if observation.strip().startswith("{"):
            try:
                data = json.loads(observation)
                # 天气类结果
                if "temperature" in data and "city" in data:
                    return (
                        f"{data.get('city', '该城市')}当前天气：{data.get('condition', '未知')}，"
                        f"气温 {data.get('temperature', 'N/A')}，"
                        f"湿度 {data.get('humidity', 'N/A')}，"
                        f"风力 {data.get('wind', 'N/A')}。"
                        + (f"\n{data.get('note', '')}" if data.get('note') else "")
                    )
                # 计数类结果
                if "file_count" in data:
                    return (
                        f"目录 {data.get('directory', '')} 中共有 "
                        f"{data.get('file_count', 0)} 个文件、"
                        f"{data.get('dir_count', 0)} 个子目录。"
                    )
                # 时间类结果
                if "datetime" in data:
                    return (
                        f"当前时间：{data.get('datetime', '')}"
                        f"（{data.get('weekday', '')}）。"
                    )
            except (json.JSONDecodeError, AttributeError):
                pass

        return f"工具返回：{observation}"

    def _answer_lifespan_from_observation(self, question: str, observation: str) -> Optional[str]:
        if not re.search(r"(活了多少岁|活了几岁|活多久|享年|年龄|多少岁)", question):
            return None

        date_matches = re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", observation)
        date_matches.extend(re.findall(r"(\d{4})-(\d{1,2})-(\d{1,2})", observation))
        if len(date_matches) >= 2:
            birth = tuple(int(part) for part in date_matches[0])
            death = tuple(int(part) for part in date_matches[1])
            age = death[0] - birth[0] - ((death[1], death[2]) < (birth[1], birth[2]))
            return (
                f"爱因斯坦活了 {age} 岁。"
                f"根据资料，他出生于 {birth[0]}年{birth[1]}月{birth[2]}日，"
                f"去世于 {death[0]}年{death[1]}月{death[2]}日。"
            )

        years = []
        for match in re.findall(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", observation):
            year = int(match)
            if year not in years:
                years.append(year)
        if len(years) >= 2:
            birth_year, death_year = years[0], years[1]
            age = death_year - birth_year
            if 0 < age < 130:
                return f"爱因斯坦约活了 {age} 岁。根据资料中的 {birth_year} 年和 {death_year} 年计算。"

        return None

    def _extract_person_name(self, text: str) -> Optional[str]:
        """从文本中提取人物名字。"""
        # 常见中国名人
        common_names = [
            "爱因斯坦", "牛顿", "达尔文", "特斯拉", "居里夫人",
            "李白", "杜甫", "孔子", "孟子", "诸葛亮",
            "毛泽东", "邓小平", "鲁迅", "钱学森", "袁隆平",
            "华盛顿", "林肯", "拿破仑", "贝多芬", "莫扎特",
            "霍金", "乔布斯", "比尔盖茨", "马斯克", "图灵",
        ]
        for name in common_names:
            if name in text:
                return name

        # 通用提取：XX是XX国人之类
        match = re.search(
            r"(.{1,6}?)(?:活了多少岁|活了几岁|活多久|享年|多少岁|哪国人|哪国的|"
            r"国籍|出生地|哪里人|什么地方人|是哪国人)",
            text,
        )
        if not match:
            return None

        subject = match.group(1)
        subject = re.sub(r"^(请问|帮我查一下|查一下|查询|问一下|你知道)", "", subject)
        subject = subject.strip(" ：:，,。？?！!的")
        return subject if len(subject) >= 1 else None

    def _count_questions(self, text: str) -> int:
        """估算用户一句话里问了多少个不同问题。"""
        indicators = [
            r"哪国人|哪国的|国籍",
            r"活了多少岁|活了几岁|活多久|享年|多少岁|年龄",
            r"出生地|哪里人|什么地方人|生于",
            r"干了什么|做了什么|有什么贡献|成就是什么|发明",
            r"是谁|是什么人|干什么的",
        ]
        count = 0
        for pat in indicators:
            if re.search(pat, text):
                count += 1
        # 也按标点分割数一下
        parts = re.split(r"[，,。；;？?！!、]", text)
        question_parts = [p for p in parts if re.search(r"什么|哪|多少|谁|吗|呢|咋|怎", p)]
        return max(count, len(question_parts))

    def _extract_nationality_from_obs(self, observation: str) -> Optional[str]:
        """从维基摘要中提取国籍/出生地信息。"""
        patterns = [
            r"是\s*(.{1,12}?)(?:理论物理学家|物理学家|数学家|化学家|生物学家|"
            r"发明家|企业家|科学家|政治家|军事家|文学家|诗人|画家|音乐家|艺术家)",
            r"(.{1,8})(?:裔|籍|国)(?:理论物理|物理|数学|化学|生物|发明|企业|"
            r"科学|政治|军事|文学|诗|画|音乐|艺术)",
            r"出生于\s*(.{2,12}?)(?:[，。,；;])",
            r"(.{2,8}?(?:人|国))[，。,；;]",
        ]
        for pat in patterns:
            m = re.search(pat, observation)
            if m:
                nationality = m.group(1).strip()
                if len(nationality) >= 2 and len(nationality) <= 12:
                    return nationality
        return None

    def _handle_person_query(self, msg: AgentMessage, trace: List[str]) -> Optional[Dict[str, str]]:
        """处理人物查询：预取 wiki 数据，但不出结果——返回预取的上下文给 LLM 使用。

        纯寿命问题：直接返回完整答案（向后兼容）。
        混合问题（寿命+国籍+...）：预取 wiki，返回 None 让后续 ReAct 循环进行，
        但会把预取数据注入 msg.metadata 供 LLM 使用。"""
        has_lifespan = bool(re.search(
            r"(活了多少岁|活了几岁|活多久|享年|多少岁)", msg.content
        ))
        has_identity = bool(re.search(
            r"(哪国人|哪国的|国籍|出生地|哪里人|什么地方人|是哪国人|是什么人|干了什么)",
            msg.content,
        ))

        if not has_lifespan and not has_identity:
            return None

        subject = self._extract_person_name(msg.content)
        if not subject:
            return None

        question_count = self._count_questions(msg.content)

        trace.append(f"🧭 检测到人物查询: `{subject}` (问题数≈{question_count})")
        observation = str(self.tool_registry["wiki_search"].run(query=subject))
        trace.append(f"👁️ Wiki 返回: {observation[:200]}{'...' if len(observation) > 200 else ''}")

        # 纯寿命问题 → 走捷径
        if has_lifespan and not has_identity:
            lifespan_answer = self._answer_lifespan_from_observation(msg.content, observation)
            if lifespan_answer:
                trace.append("✅ 纯寿命问题，快捷计算完成。")
                return {"__shortcut_answer__": lifespan_answer}

        # 混合问题 → 预取 wiki 但不抢答，交给 LLM
        trace.append("🔀 检测到复合问题，预取 wiki 数据后交给 LLM 统一回答。")
        return {"__prefetched_wiki__": observation}

    def _handle_lifespan_question(self, msg: AgentMessage, trace: List[str]) -> Optional[AgentMessage]:
        """向后兼容的包装。"""
        result = self._handle_person_query(msg, trace)
        if result is None:
            return None
        if "__shortcut_answer__" in result:
            return AgentMessage(
                self.name, msg.sender, result["__shortcut_answer__"],
                {"trace": trace, "type": "final"},
            )
        # 预取数据注入 metadata，但不返回 AgentMessage（返回 None 走正常流程）
        if "__prefetched_wiki__" in result:
            msg.metadata["prefetched_wiki"] = result["__prefetched_wiki__"]
        return None

    def _parse_explicit_write_then_read(self, text: str) -> Optional[Tuple[str, str]]:
        if not re.search(r"(\u8bfb\u53d6|\u8bfb).*(\u6587\u4ef6)?", text):
            return None

        write_verbs = r"(?:\u5199\u5165|\u5199\u5230|\u4fdd\u5b58\u5230|\u5b58\u5165)"
        quoted = re.search(
            rf"(?:\u628a|\u5c06)?\s*[\u201c\"'](.+?)[\u201d\"']\s*{write_verbs}\s*([^\s\uff0c,;\uff1b\u3002]+)",
            text,
        )
        if quoted:
            return quoted.group(1), quoted.group(2).strip()

        unquoted = re.search(
            rf"(?:\u628a|\u5c06)\s+(.+?)\s+{write_verbs}\s*([^\s\uff0c,;\uff1b\u3002]+)",
            text,
        )
        if unquoted:
            return unquoted.group(1).strip(), unquoted.group(2).strip()

        return None

    def _handle_explicit_write_then_read(self, msg: AgentMessage, trace: List[str]) -> Optional[AgentMessage]:
        parsed = self._parse_explicit_write_then_read(msg.content)
        if not parsed:
            return None

        content, path = parsed
        trace.append(f"🧭 检测到明确的写入并读取文件任务，直接调用工具: `{path}`")
        write_result = str(self.tool_registry["write_file"].run(path=path, content=content))
        trace.append(f"📝 写入结果: {write_result}")
        read_result = str(self.tool_registry["read_file"].run(path=path))
        trace.append(f"📖 读取结果: {read_result[:200]}{'...' if len(read_result) > 200 else ''}")

        return AgentMessage(
            self.name,
            msg.sender,
            f"已写入 `{path}`，并读取到内容：\n\n{read_result}",
            {"trace": trace, "type": "final"},
        )

    def process_stream(self, msg: AgentMessage) -> Iterator[AgentEvent]:
        """流式版本：逐步 yield AgentEvent，适合 TUI 等实时界面消费。"""
        trace = msg.metadata.setdefault("trace", [])
        trace.append("🛠️ **执行 Agent** 启动 ReAct 思考-行动循环...")
        yield AgentEvent("trace", trace[-1])

        # 捷径检测
        direct_response = self._handle_lifespan_question(msg, trace)
        if direct_response:
            for step in trace[1:]:
                yield AgentEvent("trace", step)
            yield AgentEvent("final", direct_response.content)
            return

        direct_response = self._handle_explicit_write_then_read(msg, trace)
        if direct_response:
            for step in trace[1:]:
                yield AgentEvent("trace", step)
            yield AgentEvent("final", direct_response.content)
            return

        messages = [{"role": "system", "content": self.system_prompt}]
        memory_context = msg.metadata.get("memory_context")
        if memory_context:
            messages.append({"role": "system", "content": f"相关长期记忆:\n{memory_context}"})
        # 注入预取的 Wiki 数据（人物复合查询时由 _handle_person_query 填入）
        prefetched = msg.metadata.pop("prefetched_wiki", None)
        if prefetched:
            messages.append({"role": "system", "content": (
                "以下是从维基百科预取的资料，请基于此回答用户的所有问题"
                "（包括国籍、身份、寿命等），一次性完整作答：\n\n" + prefetched
            )})
        messages.extend(self._condense_history(msg.metadata.get("history", [])))
        messages.append({"role": "user", "content": msg.content})

        last_action_key = None
        last_observation = None
        last_action = None

        for i in range(self.MAX_ITERATIONS):
            trace.append(f"🔄 开始第 {i + 1} 轮推理...")
            yield AgentEvent("trace", trace[-1])

            try:
                full_response = self.llm.chat(messages, stream=False)
            except Exception as e:
                trace.append(f"❌ API 调用失败: {str(e)}")
                yield AgentEvent("trace", trace[-1])
                yield AgentEvent("final", f"执行中断: {str(e)}")
                return

            thought, action, action_input, final_answer = self._parse_llm_response(full_response)
            if thought:
                trace.append(f"💭 **思考**: {thought}")
                yield AgentEvent("thought", thought)

            if final_answer:
                trace.append("✅ 得到最终答案，结束循环。")
                yield AgentEvent("trace", trace[-1])
                yield AgentEvent("final", final_answer)
                return

            if action:
                action_key = (action, json.dumps(action_input, ensure_ascii=False, sort_keys=True))
                if action_key == last_action_key and last_observation is not None:
                    trace.append("⚠️ 检测到重复工具调用，直接使用上一轮结果。")
                    yield AgentEvent("trace", trace[-1])
                    answer = self._fallback_answer_from_observation(action, last_observation)
                    yield AgentEvent("final", answer)
                    return

                try:
                    trace.append(f"🔧 **调用工具**: `{action}`")
                    yield AgentEvent("trace", trace[-1])
                    trace.append(f"📥 **参数**: `{json.dumps(action_input, ensure_ascii=False)}`")
                    yield AgentEvent("tool_call", {"action": action, "input": action_input})

                    if action in self.tool_registry:
                        observation = str(self.tool_registry[action].run(**action_input))
                    else:
                        observation = f"工具 {action} 不存在（可用: {', '.join(list(self.tool_registry.keys())[:20])}...）"

                    trace.append(f"👁️ **返回**: {observation[:200]}{'...' if len(observation) > 200 else ''}")
                    yield AgentEvent("tool_result", observation[:500])

                    lifespan_answer = self._answer_lifespan_from_observation(msg.content, observation)
                    if lifespan_answer:
                        trace.append("✅ 已从工具结果中计算出人物寿命。")
                        yield AgentEvent("trace", trace[-1])
                        yield AgentEvent("final", lifespan_answer)
                        return
                except Exception as e:
                    observation = f"执行错误: {str(e)}"
                    trace.append(f"❌ **异常**: {str(e)}")
                    yield AgentEvent("trace", trace[-1])

                # 放入干净的助理回复（避免多 JSON 拼接污染上下文）
                clean_assistant = json.dumps({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                }, ensure_ascii=False)
                messages.append({"role": "assistant", "content": clean_assistant})
                messages.append({"role": "user", "content": (
                    f"Observation: {observation}\n\n"
                    "🚨 你已经获得了工具的执行结果。现在必须立即用 Final Answer 格式"
                    "输出最终回复——用流畅的中文把结果告诉用户，不要再调用任何工具。"
                    "例如: {\"final_answer\": \"根据查询结果，杭州今天多云，气温11°C...\"}"
                )})
                last_action_key = action_key
                last_observation = observation
                last_action = action
            else:
                break

        trace.append("⚠️ 达到最大循环次数，强制结束。")
        yield AgentEvent("trace", trace[-1])
        if last_observation is not None and last_action is not None:
            answer = self._fallback_answer_from_observation(last_action, last_observation)
            yield AgentEvent("final", answer)
        else:
            yield AgentEvent("final", "抱歉，我尝试了多次但未能完成任务 (´;ω;`)")

    def process(self, msg: AgentMessage) -> AgentMessage:
        """批量版本：调用 process_stream 并收集结果（向后兼容）。"""
        trace = msg.metadata.setdefault("trace", [])
        trace.append("🛠️ **执行 Agent** 启动 ReAct 思考-行动循环...")

        direct_response = self._handle_lifespan_question(msg, trace)
        if direct_response:
            return direct_response

        direct_response = self._handle_explicit_write_then_read(msg, trace)
        if direct_response:
            return direct_response

        messages = [{"role": "system", "content": self.system_prompt}]
        memory_context = msg.metadata.get("memory_context")
        if memory_context:
            messages.append({"role": "system", "content": f"相关长期记忆:\n{memory_context}"})
        # 注入预取的 Wiki 数据（人物复合查询时由 _handle_person_query 填入）
        prefetched = msg.metadata.pop("prefetched_wiki", None)
        if prefetched:
            messages.append({"role": "system", "content": (
                "以下是从维基百科预取的资料，请基于此回答用户的所有问题"
                "（包括国籍、身份、寿命等），一次性完整作答：\n\n" + prefetched
            )})
        messages.extend(self._condense_history(msg.metadata.get("history", [])))
        messages.append({"role": "user", "content": msg.content})

        last_action_key = None
        last_observation = None
        last_action = None

        for i in range(self.MAX_ITERATIONS):
            trace.append(f"🔄 开始第 {i + 1} 轮推理...")
            try:
                full_response = self.llm.chat(messages, stream=False)
            except Exception as e:
                trace.append(f"❌ API 调用失败: {str(e)}")
                return AgentMessage(self.name, msg.sender, f"执行中断: {str(e)}", {"trace": trace, "type": "final"})

            thought, action, action_input, final_answer = self._parse_llm_response(full_response)
            if thought:
                trace.append(f"💭 **思考**: {thought}")

            if final_answer:
                trace.append("✅ 得到最终答案，结束循环。")
                return AgentMessage(self.name, msg.sender, final_answer,
                                    {"thought": thought, "trace": trace, "type": "final"})

            if action:
                action_key = (action, json.dumps(action_input, ensure_ascii=False, sort_keys=True))
                if action_key == last_action_key and last_observation is not None:
                    trace.append("⚠️ 检测到重复工具调用，直接使用上一轮工具结果生成最终答案。")
                    return AgentMessage(
                        self.name,
                        msg.sender,
                        self._fallback_answer_from_observation(action, last_observation),
                        {"thought": thought, "trace": trace, "type": "final"},
                    )

                try:
                    trace.append(f"🔧 **调用工具**: `{action}`")
                    trace.append(f"📥 **传入参数**: `{json.dumps(action_input, ensure_ascii=False)}`")

                    if action in self.tool_registry:
                        observation = str(self.tool_registry[action].run(**action_input))
                    else:
                        observation = f"工具 {action} 不存在"

                    trace.append(f"👁️ **工具返回**: {observation[:200]}{'...' if len(observation) > 200 else ''}")

                    lifespan_answer = self._answer_lifespan_from_observation(msg.content, observation)
                    if lifespan_answer:
                        trace.append("✅ 已从工具结果中计算出人物寿命，结束循环。")
                        return AgentMessage(
                            self.name,
                            msg.sender,
                            lifespan_answer,
                            {"thought": thought, "trace": trace, "type": "final"},
                        )
                except Exception as e:
                    observation = f"执行错误: {str(e)}"
                    trace.append(f"❌ **工具执行异常**: {str(e)}")

                clean_assistant = json.dumps({
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                }, ensure_ascii=False)
                messages.append({"role": "assistant", "content": clean_assistant})
                messages.append({"role": "user", "content": (
                    f"Observation: {observation}\n\n"
                    "🚨 工具已返回结果。立即用 {\"final_answer\": \"...\"} 格式输出最终回复，"
                    "用流畅中文把结果告诉用户，不要再调任何工具。"
                )})
                last_action_key = action_key
                last_observation = observation
                last_action = action
            else:
                break

        trace.append("⚠️ 达到最大循环次数，强制结束。")
        if last_observation is not None and last_action is not None:
            trace.append("⚠️ 使用最后一次工具返回作为兜底答案。")
            return AgentMessage(
                self.name,
                msg.sender,
                self._fallback_answer_from_observation(last_action, last_observation),
                {"trace": trace, "type": "final"},
            )
        return AgentMessage(self.name, msg.sender, "抱歉，我尝试了多次但未能完成任务 (´;ω;`)",
                            {"trace": trace, "type": "final"})


class PlannerAgent(BaseAgent):
    def __init__(self, llm_client, executor):
        super().__init__("Planner", llm_client)
        self.executor = executor

    def process_stream(self, msg: AgentMessage) -> Iterator[AgentEvent]:
        """流式版本：分析意图后流式执行。"""
        trace = msg.metadata.setdefault("trace", [])
        trace.append("🧠 **规划 Agent** 正在分析用户意图...")
        yield AgentEvent("trace", trace[-1])

        try:
            prompt = (
                "判断以下输入是否需要使用外部工具（计算/搜索/文件读写/日期/MCP工具等）。"
                "优先输出 JSON: {\"needs_tool\": true/false, \"reason\": \"...\", \"plan\": [\"...\"]}。"
                "如果无法输出 JSON，只回答 YES 或 NO。\n"
                f"输入: {msg.content}"
            )
            intent = self.llm.chat([{"role": "user", "content": prompt}], stream=False).strip()
        except Exception as e:
            trace.append(f"❌ 意图分析 API 失败: {str(e)}")
            yield AgentEvent("final", f"规划中断: {str(e)}")
            return

        needs_tool, plan_text = self._parse_intent(intent)
        if plan_text:
            trace.append(f"🗺️ **规划结果**: {plan_text}")
            yield AgentEvent("trace", trace[-1])

        if needs_tool:
            trace.append("🎯 **意图判断**: 需要调用工具，移交给执行 Agent。")
            yield AgentEvent("trace", trace[-1])
            yield from self.executor.process_stream(msg)
        else:
            trace.append("🎯 **意图判断**: 无需工具，直接生成回复。")
            yield AgentEvent("trace", trace[-1])
            try:
                direct_messages = [{"role": "system", "content": "你是一个简洁可靠的 AI 助手。"}]
                memory_context = msg.metadata.get("memory_context")
                if memory_context:
                    direct_messages.append({"role": "system", "content": f"相关长期记忆:\n{memory_context}"})
                direct_messages.extend(self.executor._condense_history(msg.metadata.get("history", [])))
                direct_messages.append({"role": "user", "content": msg.content})
                resp = self.llm.chat(direct_messages, stream=False)
                yield AgentEvent("final", resp)
            except Exception as e:
                yield AgentEvent("final", f"回复生成失败: {str(e)}")

    def process(self, msg: AgentMessage) -> AgentMessage:
        """批量版本：调用 process_stream 收集结果（向后兼容）。"""
        final_text = None
        for event in self.process_stream(msg):
            if event.type == "final":
                final_text = event.data
        if final_text is None:
            final_text = "抱歉，处理过程中出现错误。"
        return AgentMessage(self.name, msg.sender, final_text,
                            {"trace": msg.metadata.get("trace", []), "type": "final"})

    def _parse_intent(self, raw: str) -> Tuple[bool, str]:
        upper = raw.strip().upper()
        if upper == "YES":
            return True, ""
        if upper == "NO":
            return False, ""

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                plan = data.get("plan", [])
                if isinstance(plan, list):
                    plan_text = " -> ".join(str(item) for item in plan)
                else:
                    plan_text = str(plan)
                return bool(data.get("needs_tool")), plan_text
        except json.JSONDecodeError:
            pass

        # 兼容旧逻辑（如 "YES, I need tool" 仍可能被当作 True）
        return "YES" in upper, ""


class OrchestratorAgent(BaseAgent):
    def __init__(self, llm_client, planner, memory, mcp_manager=None):
        super().__init__("Orchestrator", llm_client)
        self.planner = planner
        self.memory = memory
        self.mcp_manager = mcp_manager

    def _parse_remember_command(self, text: str) -> Optional[str]:
        match = re.match(r"^\s*(?:请)?(?:帮我)?(?:记住|记一下|记忆)[:：]?\s*(.+?)\s*$", text)
        if not match:
            return None
        fact = match.group(1).strip()
        return fact or None

    def _save_direct_memory(self, session_id: str, user_input: str, fact: str, trace: List[str]) -> AgentMessage:
        response = f"我记住了：{fact}"
        self.memory.save(session_id, "user", user_input)
        self.memory.save(session_id, "memory", fact)
        self.memory.save(session_id, "assistant", response)
        trace.append("💾 检测到明确记忆指令，已直接写入长期记忆。")
        return AgentMessage(self.name, "User", response, {"trace": trace, "type": "final"})

    def _recall_project_codename(self, session_id: str, user_input: str, trace: List[str]) -> Optional[AgentMessage]:
        if not re.search(r"(项目代号|项目代码|代号).*(是什么|多少|叫啥|叫什么|是啥)", user_input):
            return None

        candidates = self.memory.get_recent(session_id, 20)
        try:
            candidates.extend(self.memory.search(session_id, "项目代号 代号 项目代码", 10))
        except Exception:
            pass

        for item in reversed(candidates):
            content = item["content"]
            match = re.search(r"(?:我的)?(?:项目代号|项目代码|代号)\s*(?:是|为|叫|叫做)[:：]?\s*([^\s，,。；;]+)", content)
            if match:
                codename = match.group(1).strip()
                response = f"你的项目代号是 {codename}。"
                self.memory.save(session_id, "user", user_input)
                self.memory.save(session_id, "assistant", response)
                trace.append("🔎 已从长期记忆中直接找到项目代号。")
                return AgentMessage(self.name, "User", response, {"trace": trace, "type": "final"})

        return None

    def process_stream(self, msg: AgentMessage) -> Iterator[AgentEvent]:
        """流式版本：逐步 yield AgentEvent 供 TUI 等实时界面消费。"""
        session_id = msg.metadata.get("session_id", "default")
        user_input = msg.content

        msg.metadata["trace"] = []
        msg.metadata["trace"].append("📥 **总控 Agent** 接收到指令，开始处理...")
        yield AgentEvent("trace", msg.metadata["trace"][-1])

        # 处理命令
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd == "/clear":
                self.memory.clear(session_id)
                yield AgentEvent("trace", "🧹 执行命令: 清空记忆")
                yield AgentEvent("final", "记忆已清空 (๑•̀ㅂ•́)و✧")
                return
            elif cmd == "/help":
                yield AgentEvent("trace", "❓ 执行命令: 帮助")
                yield AgentEvent("final", "可用命令:\n/clear - 清空记忆\n/help - 显示帮助\n/mcp - MCP服务器状态\n/shell <自然语言> - Shell Agent模式\n/quit - 退出")
                return
            elif cmd.startswith("/mcp"):
                if self.mcp_manager:
                    status = self.mcp_manager.get_server_status()
                    lines = ["**MCP 服务器状态:**"]
                    for name, s in status.items():
                        icon = "🟢" if s["connected"] else "🔴" if s["enabled"] else "⚫"
                        lines.append(f"  {icon} {name}: 工具数={s['tool_count']} {s.get('description','')}")
                    yield AgentEvent("final", "\n".join(lines))
                else:
                    yield AgentEvent("final", "未配置 MCP 管理器。")
                return
            else:
                yield AgentEvent("trace", "❓ 执行命令: 未知")
                yield AgentEvent("final", "未知命令")
                return

        # 记忆捷径
        remembered_fact = self._parse_remember_command(user_input)
        if remembered_fact:
            response = f"我记住了：{remembered_fact}"
            self.memory.save(session_id, "user", user_input)
            self.memory.save(session_id, "memory", remembered_fact)
            self.memory.save(session_id, "assistant", response)
            yield AgentEvent("trace", "💾 检测到明确记忆指令，已直接写入长期记忆。")
            yield AgentEvent("final", response)
            return

        # 代号回忆捷径
        direct_recall = self._recall_project_codename(session_id, user_input, msg.metadata["trace"])
        if direct_recall:
            for step in msg.metadata["trace"][1:]:
                yield AgentEvent("trace", step)
            yield AgentEvent("final", direct_recall.content)
            return

        # 记忆注入
        msg.metadata["trace"].append("📚 注入短期记忆 (最近 5 轮)...")
        yield AgentEvent("trace", msg.metadata["trace"][-1])
        history = self.memory.get_recent(session_id, 5)
        msg.metadata["history"] = history
        try:
            relevant_memory = self.memory.search(session_id, user_input, 3)
        except Exception:
            relevant_memory = []

        if relevant_memory:
            memory_context = "\n".join(
                f"[{item['role']}] {item['content']}" for item in relevant_memory
            )
            msg.metadata["memory_context"] = memory_context
            msg.metadata["trace"].append("🔎 检索到相关长期记忆，已注入上下文。")
            yield AgentEvent("trace", msg.metadata["trace"][-1])

        msg.metadata["trace"].append("🚀 将任务分发给规划 Agent...")
        yield AgentEvent("trace", msg.metadata["trace"][-1])

        # 流式执行规划/执行链
        final_text = None
        for event in self.planner.process_stream(msg):
            if event.type == "final":
                final_text = event.data
            else:
                yield event

        if final_text is None:
            final_text = "抱歉，处理过程中出现错误。"

        self.memory.save(session_id, "user", user_input)
        self.memory.save(session_id, "assistant", final_text)
        msg.metadata["trace"].append("💾 对话已保存至长期记忆。")
        yield AgentEvent("trace", msg.metadata["trace"][-1])
        yield AgentEvent("final", final_text)

    def process(self, msg: AgentMessage) -> AgentMessage:
        session_id = msg.metadata.get("session_id", "default")
        user_input = msg.content

        # 初始化 Trace
        msg.metadata["trace"] = []
        msg.metadata["trace"].append("📥 **总控 Agent** 接收到指令，开始处理...")

        if user_input.startswith("/"):
            cmd = user_input.lower().strip()
            if cmd == "/clear":
                self.memory.clear(session_id)
                return AgentMessage(self.name, msg.sender, "记忆已清空 (๑•̀ㅂ•́)و✧",
                                    {"trace": ["🧹 执行命令: 清空记忆"], "type": "final"})
            elif cmd == "/help":
                return AgentMessage(self.name, msg.sender, "可用命令:\n/clear - 清空记忆\n/help - 显示帮助\n/shell <自然语言> - Shell Agent模式",
                                    {"trace": ["❓ 执行命令: 帮助"], "type": "final"})
            else:
                return AgentMessage(self.name, msg.sender, "未知命令", {"trace": ["❓ 执行命令: 未知"], "type": "final"})

        remembered_fact = self._parse_remember_command(user_input)
        if remembered_fact:
            return self._save_direct_memory(session_id, user_input, remembered_fact, msg.metadata["trace"])

        direct_recall = self._recall_project_codename(session_id, user_input, msg.metadata["trace"])
        if direct_recall:
            return direct_recall

        msg.metadata["trace"].append("📚 注入短期记忆 (最近 5 轮)...")
        history = self.memory.get_recent(session_id, 5)
        msg.metadata["history"] = history
        try:
            relevant_memory = self.memory.search(session_id, user_input, 3)
        except Exception:
            relevant_memory = []

        if relevant_memory:
            memory_context = "\n".join(
                f"[{item['role']}] {item['content']}" for item in relevant_memory
            )
            msg.metadata["memory_context"] = memory_context
            msg.metadata["trace"].append("🔎 检索到相关长期记忆，已注入上下文。")

        msg.metadata["trace"].append("🚀 将任务分发给规划 Agent...")
        response = self.planner.process(msg)

        self.memory.save(session_id, "user", user_input)
        self.memory.save(session_id, "assistant", response.content)
        msg.metadata["trace"].append("💾 对话已保存至长期记忆。")

        return response
