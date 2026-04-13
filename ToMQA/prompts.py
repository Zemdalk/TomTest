"""ToMQA prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": (
        "Read the story and answer the question.\n"
        "Answer with a single word or short phrase.\n\n"
        "Story: {story}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
}


def get_template(method: str) -> str:
    """获取指定方法的 prompt 模板"""
    return PROMPTS.get(method, PROMPTS["zero_shot"])


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """构建 prompt"""
    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    story = story_info.get("full_story", "") or ""

    question = row.get("Question", "") or ""

    return template.format(
        story=story,
        question=question,
    )
