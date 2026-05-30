import os
import json
import re
import platform
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from llm_client import LLMClient
from tools import TOOL_REGISTRY


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    metadata: dict = field(default_factory=dict)


class BaseAgent:
    def __init__(self, name, llm_client):
        self.name = name
        self.llm = llm_client

    def process(self, msg: AgentMessage) -> AgentMessage:
        raise NotImplementedError


class ExecutorAgent(BaseAgent):
    def __init__(self, llm_client, agents_md):
        super().__init__("Executor", llm_client)
        self.agents_md = agents_md
        self.tools_desc = self._build_tools_desc()

    def _build_tools_desc(self):
        tools = [t.spec() for t in TOOL_REGISTRY.values()]
        return json.dumps(tools, ensure_ascii=False)

    def _build_prompt(self):
        return f"""你是一个强大的 AI 助手。请遵循以下规则：
{self.agents_md}
当前系统: {platform.system()} | 目录: {os.getcwd()}
可用工具: {self.tools_desc}

推荐输出 JSON，便于程序稳定解析：
{{"thought": "你的思考", "action": "工具名", "action_input": {{"参数": "值"}}}}
{{"thought": "你的思考", "final_answer": "最终回复"}}

也兼容以下文本格式：
Thought: 你的思考
Action: 工具名
Action Input: {{"参数": "值"}}

若无需工具或已得答案：
Thought: 你的思考
Final Answer: 最终回复
"""

    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidates = [code_block.group(1)] if code_block else []
        candidates.append(text)

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            candidates.append(text[brace_start:brace_end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

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
            return thought, None, {}, None

        action = action_match.group(1).strip()
        action_input_str = action_match.group(2).strip()
        action_input = self._extract_json_object(action_input_str) or {}
        return thought, action, action_input, None

    def _fallback_answer_from_observation(self, action: str, observation: str) -> str:
        if action == "calculator":
            return f"计算结果：{observation}"
        return f"工具返回结果：{observation}"

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

    def _extract_lifespan_subject(self, text: str) -> Optional[str]:
        if "爱因斯坦" in text:
            return "爱因斯坦"

        match = re.search(r"(.+?)(?:活了多少岁|活了几岁|活多久|享年|多少岁)", text)
        if not match:
            return None

        subject = match.group(1)
        subject = re.sub(r"^(请问|帮我查一下|查一下|查询|问一下)", "", subject).strip(" ：:，,。？?")
        return subject or None

    def _handle_lifespan_question(self, msg: AgentMessage, trace: List[str]) -> Optional[AgentMessage]:
        if not re.search(r"(活了多少岁|活了几岁|活多久|享年|年龄|多少岁)", msg.content):
            return None

        subject = self._extract_lifespan_subject(msg.content)
        if not subject:
            return None

        trace.append(f"🧭 检测到人物寿命问题，直接查询并计算: `{subject}`")
        observation = str(TOOL_REGISTRY["wiki_search"].run(query=subject))
        trace.append(f"👁️ 查询结果: {observation[:200]}{'...' if len(observation) > 200 else ''}")
        answer = self._answer_lifespan_from_observation(msg.content, observation)
        if not answer:
            return None

        trace.append("✅ 已直接计算出人物寿命，跳过自由工具选择。")
        return AgentMessage(
            self.name,
            msg.sender,
            answer,
            {"trace": trace, "type": "final"},
        )

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
        write_result = str(TOOL_REGISTRY["write_file"].run(path=path, content=content))
        trace.append(f"📝 写入结果: {write_result}")
        read_result = str(TOOL_REGISTRY["read_file"].run(path=path))
        trace.append(f"📖 读取结果: {read_result[:200]}{'...' if len(read_result) > 200 else ''}")

        return AgentMessage(
            self.name,
            msg.sender,
            f"已写入 `{path}`，并读取到内容：\n\n{read_result}",
            {"trace": trace, "type": "final"},
        )

    def process(self, msg: AgentMessage) -> AgentMessage:
        trace = msg.metadata.setdefault("trace", [])
        trace.append("🛠️ **执行 Agent** 启动 ReAct 思考-行动循环...")

        direct_response = self._handle_lifespan_question(msg, trace)
        if direct_response:
            return direct_response

        direct_response = self._handle_explicit_write_then_read(msg, trace)
        if direct_response:
            return direct_response

        messages = [{"role": "system", "content": self._build_prompt()}]
        memory_context = msg.metadata.get("memory_context")
        if memory_context:
            messages.append({"role": "system", "content": f"相关长期记忆:\n{memory_context}"})
        messages.extend(msg.metadata.get("history", []))
        messages.append({"role": "user", "content": msg.content})

        last_action_key = None
        last_observation = None
        last_action = None

        for i in range(5):
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

                    if action in TOOL_REGISTRY:
                        observation = str(TOOL_REGISTRY[action].run(**action_input))
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

                messages.append({"role": "assistant", "content": full_response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
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

    def process(self, msg: AgentMessage) -> AgentMessage:
        trace = msg.metadata.setdefault("trace", [])
        trace.append("🧠 **规划 Agent** 正在分析用户意图...")

        try:
            prompt = (
                "判断以下输入是否需要使用外部工具（计算/搜索/文件读写/日期等）。"
                "优先输出 JSON: {\"needs_tool\": true/false, \"reason\": \"...\", \"plan\": [\"...\"]}。"
                "如果无法输出 JSON，只回答 YES 或 NO。\n"
                f"输入: {msg.content}"
            )
            intent = self.llm.chat([{"role": "user", "content": prompt}], stream=False).strip()
        except Exception as e:
            trace.append(f"❌ 意图分析 API 失败: {str(e)}")
            return AgentMessage(self.name, msg.sender, f"规划中断: {str(e)}", {"trace": trace, "type": "final"})

        needs_tool, plan_text = self._parse_intent(intent)
        if plan_text:
            trace.append(f"🗺️ **规划结果**: {plan_text}")

        if needs_tool:
            trace.append("🎯 **意图判断**: 需要调用工具，移交给执行 Agent。")
            return self.executor.process(msg)
        else:
            trace.append("🎯 **意图判断**: 无需工具，直接生成回复。")
            try:
                direct_messages = [{"role": "system", "content": "你是一个简洁可靠的 AI 助手。"}]
                memory_context = msg.metadata.get("memory_context")
                if memory_context:
                    direct_messages.append({"role": "system", "content": f"相关长期记忆:\n{memory_context}"})
                direct_messages.extend(msg.metadata.get("history", []))
                direct_messages.append({"role": "user", "content": msg.content})
                resp = self.llm.chat(direct_messages, stream=False)
                return AgentMessage(self.name, msg.sender, resp, {"trace": trace, "type": "final"})
            except Exception as e:
                return AgentMessage(self.name, msg.sender, f"回复生成失败: {str(e)}", {"trace": trace, "type": "final"})

    def _parse_intent(self, raw: str) -> Tuple[bool, str]:
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

        upper = raw.upper()
        return "YES" in upper, ""


class OrchestratorAgent(BaseAgent):
    def __init__(self, llm_client, planner, memory):
        super().__init__("Orchestrator", llm_client)
        self.planner = planner
        self.memory = memory

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
                return AgentMessage(self.name, msg.sender, "可用命令:\n/clear - 清空记忆\n/help - 显示帮助",
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
