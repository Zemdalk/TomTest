"""ToMi 数据集的输出 schema"""
from pydantic import BaseModel, constr


class OneWordAnswer(BaseModel):
    """单词答案 schema（不允许空白字符）"""
    answer: constr(strip_whitespace=True, min_length=1, pattern=r"^\S+$")


# schema 字典：config.yaml 中引用主 schema，其他 schema 供内部调用
SCHEMAS = {
    "OneWordAnswer": OneWordAnswer,
}
