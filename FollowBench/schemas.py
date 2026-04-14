"""FollowBench 输出 schema"""
from pydantic import BaseModel
from typing import Literal


class OpenAnswer(BaseModel):
    """开放式生成答案 schema"""
    answer: str


class JudgeAnswer(BaseModel):
    """LLM Judge 约束满足性判断 schema（内部调用）"""
    answer: Literal["YES", "NO"]


SCHEMAS = {
    "OpenAnswer": OpenAnswer,
    "JudgeAnswer": JudgeAnswer,
}
