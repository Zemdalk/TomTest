"""ToMi prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

Return the answer JSON using exactly one lowercase word in the `answer` field.
Do not output a sentence, explanation, or punctuation.""",
    "cot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

Think step by step internally, then return the answer JSON using exactly one lowercase word in the `answer` field.
Do not output a sentence, explanation, or punctuation.""",
}


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 prompt"""
    story = row.get("instruction", "")
    question = row.get("input", "")
    return template.format(story=story, question=question)


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板"""
    return PROMPTS.get(method, PROMPTS["zero_shot"])
