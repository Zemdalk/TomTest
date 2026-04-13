"""ToMi 数据集的 metrics 计算"""
from typing import Any, Dict, List


def _normalize_word(text: Any) -> str:
    """归一化为单词比较格式。"""
    if text is None:
        return ""
    return str(text).strip().lower()


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 ToMi 的 metrics（单词答案精确匹配）"""
    gold_answers = [_normalize_word(row.get("output", "")) for row in data]
    pred_answers = [_normalize_word(p) for p in predictions]

    correct = sum(1 for p, g in zip(pred_answers, gold_answers) if p == g)
    accuracy = correct / len(pred_answers) if pred_answers else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(pred_answers),
    }
