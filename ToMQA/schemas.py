"""ToMQA 数据集的输出 schema"""
from pydantic import BaseModel


class OpenAnswer(BaseModel):
    """开放式问答 schema"""
    answer: str = ""


# schema 字典：config.yaml 中引用主 schema，其他 schema 供内部调用
SCHEMAS = {
    "OpenAnswer": OpenAnswer,
}
