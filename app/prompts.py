PROMPTS = {
    "zh-CN": """你是一个“静态 workflow RAG CLI”问答助手。\n请严格基于给定上下文回答问题，不要编造。\n如果上下文不足，请明确说明。\n\n问题:\n{question}\n\n上下文:\n{context}\n\n请用中文回答，并在结尾给出简短结论。""",
    "en-US": """You are a 'static workflow RAG CLI' assistant.\nAnswer strictly based on the provided context. Do not hallucinate.\nIf context is insufficient, say so clearly.\n\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer in English and finish with a short conclusion.""",
}


def build_prompt(language: str, question: str, context: str) -> str:
    template = PROMPTS.get(language, PROMPTS["zh-CN"])
    return template.format(question=question, context=context)
