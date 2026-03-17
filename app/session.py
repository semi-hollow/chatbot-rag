from dataclasses import dataclass

from app.config import SESSION_MAX_TURNS, SESSION_RECENT_TURNS
from app.ollama_client import chat


@dataclass
class Turn:
    question: str
    answer: str


class SessionMemory:
    def __init__(self, max_turns: int = SESSION_MAX_TURNS, recent_turns: int = SESSION_RECENT_TURNS):
        self.max_turns = max_turns
        self.recent_turns = recent_turns
        self.turns: list[Turn] = []
        self.summary = ""

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))
        if len(self.turns) > self.max_turns:
            self._compress_history()

    def recent_history_text(self) -> str:
        recent = self.turns[-self.recent_turns :]
        return "\n".join([f"Q: {t.question}\nA: {t.answer}" for t in recent])

    def full_context_text(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"Summary:\n{self.summary}")
        recent = self.recent_history_text()
        if recent:
            parts.append(f"Recent turns:\n{recent}")
        return "\n\n".join(parts)

    def rewrite_question(self, question: str) -> str:
        context = self.full_context_text()
        if not context:
            return question

        prompt = (
            "Rewrite the follow-up question into a standalone question for retrieval. "
            "Do not answer it. Return only one rewritten sentence.\n\n"
            f"Conversation context:\n{context}\n\n"
            f"Follow-up question:\n{question}"
        )
        rewritten = chat(prompt).strip()
        return rewritten or question

    def _compress_history(self) -> None:
        to_summarize = self.turns[:-self.recent_turns]
        if not to_summarize:
            return

        history = "\n".join([f"Q: {t.question}\nA: {t.answer}" for t in to_summarize])
        base = f"Existing summary:\n{self.summary}\n\n" if self.summary else ""
        prompt = (
            "Summarize conversation memory into short factual bullets for future QA context.\n"
            f"{base}History to summarize:\n{history}"
        )
        self.summary = chat(prompt).strip() or self.summary
        self.turns = self.turns[-self.recent_turns :]
