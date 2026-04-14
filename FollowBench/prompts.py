"""FollowBench prompts

FollowBench 的 instruction 字段本身即为完整 prompt，无需额外模板包装。
"""
from typing import Any, Dict


PROMPTS = {
    "zero_shot": "{instruction}",
}


def get_template(method: str) -> str:
    return PROMPTS.get(method, PROMPTS["zero_shot"])


def build_prompt(template: str, row: Dict[str, Any]) -> str:
    """直接返回 Question 字段（即当前 level 的完整 instruction）"""
    return template.format(instruction=row["Question"])
