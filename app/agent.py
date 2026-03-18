from dataclasses import dataclass, field

from app.pipeline import answer_question
from app.session import SessionMemory


@dataclass
class AgentState:
    language: str = "zh-CN"
    session: SessionMemory = field(default_factory=SessionMemory)
    steps: list[dict] = field(default_factory=list)


class Toolset:
    def retrieve_and_answer(self, question: str, **kwargs) -> dict:
        return answer_question(question=question, **kwargs)


class MinimalDocumentQAAgent:
    def __init__(self, state: AgentState | None = None):
        self.state = state or AgentState()
        self.tools = Toolset()

    def _route(self, question: str) -> str:
        short = question.strip().lower()
        if short in {"hi", "hello", "你好"}:
            return "direct"
        return "rag"

    def run(self, question: str, show_steps: bool = False, **kwargs) -> dict:
        self.state.steps = []
        route = self._route(question)
        self.state.steps.append({"step": "router", "decision": route})

        if route == "direct":
            answer = "你好，我是文档问答助手。你可以直接提问文档内容。"
            if self.state.language != "zh-CN":
                answer = "Hello, I am a document QA assistant. Ask me about your documents."
            result = {"answer": answer, "citations": [], "retrieved_chunks": [], "trace_id": "agent-direct"}
        else:
            rewritten = self.state.session.rewrite_question(question)
            self.state.steps.append(
                {"step": "planner", "action": "rewrite+retrieve", "rewritten_question": rewritten}
            )
            result = self.tools.retrieve_and_answer(
                question=question,
                rewrite_question=rewritten,
                language=self.state.language,
                trace_enabled=True,
                **kwargs,
            )

            if "证据不足" in result["answer"] or "Insufficient evidence" in result["answer"]:
                self.state.steps.append({"step": "judge", "decision": "insufficient_evidence"})
            else:
                self.state.steps.append({"step": "judge", "decision": "answer_ok"})

        self.state.session.add_turn(question, result["answer"])
        self.state.steps.append({"step": "stop", "reason": "single_cycle_complete"})
        if show_steps:
            result["agent_steps"] = list(self.state.steps)
        return result
