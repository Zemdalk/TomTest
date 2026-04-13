"""ToMQA 数据集的 metrics 计算"""
from typing import Any, Dict, List
import re


def _normalize(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()

    # remove simple leading prefixes
    for prefix in ("answer:", "answer", "ans:", "output:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break

    # strip quotes
    if (s.startswith("\"") and s.endswith("\"")) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # normalize spaces to underscores
    s = s.replace(" ", "_")

    # strip punctuation at edges (keep word chars and underscores)
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s


def normalize_answer(text: Any) -> str:
    """对答案做统一归一化（小写 + 空格转下划线 + 去首尾标点）"""
    return _normalize(text)


def _get_gold_list(row: Dict[str, Any]) -> List[str]:
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    gold = answer_block.get("Correct_Answer", [])
    if isinstance(gold, list):
        return [str(g) for g in gold]
    if gold is None:
        return []
    return [str(gold)]


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, correct: bool) -> None:
    key_str = str(key)
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if correct:
        stats[key_str]["correct"] += 1


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算 ToMQA 的 metrics

    - 整体准确率（基于标准答案列表的归一化匹配）
    - 按维度、难度、任务类型、order 分组的二级指标
    """
    correct = 0
    total = len(predictions)

    by_dimension: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}
    by_task_type: Dict[str, Dict[str, int]] = {}
    by_order: Dict[str, Dict[str, int]] = {}

    for pred, row in zip(predictions, data):
        gold_list = _get_gold_list(row)
        pred_norm = _normalize(pred)
        gold_norm = {_normalize(g) for g in gold_list}

        is_correct = bool(pred_norm) and pred_norm in gold_norm if gold_norm else False
        if is_correct:
            correct += 1

        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        dimension = meta.get("dimension", "unknown")
        if isinstance(dimension, list) and dimension:
            dimension_value = dimension[0]
        else:
            dimension_value = dimension if dimension else "unknown"

        difficulty = meta.get("difficulty", "unknown") or "unknown"
        task_type = meta.get("task_type", "unknown") or "unknown"
        order = meta.get("order", "unknown")

        _update_group(by_dimension, dimension_value, is_correct)
        _update_group(by_difficulty, difficulty, is_correct)
        _update_group(by_task_type, task_type, is_correct)
        _update_group(by_order, order, is_correct)

    accuracy = correct / total if total else 0

    # 只保留 ToMQA 关注的两类维度，并确保缺失时也有 key
    target_dimensions = ["first_order_belief", "second_order_belief"]
    for dim in target_dimensions:
        if dim not in by_dimension:
            by_dimension[dim] = {"correct": 0, "total": 0}

    def _flatten(group: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}.{k}": (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in group.items()
        }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **_flatten({k: by_dimension[k] for k in target_dimensions}, "by_dimension"),
        **_flatten(by_difficulty, "by_difficulty"),
        **_flatten(by_task_type, "by_task_type"),
        **_flatten(by_order, "by_order"),
        "by_dimension": {
            k: (by_dimension[k]["correct"] / by_dimension[k]["total"] if by_dimension[k]["total"] else 0)
            for k in target_dimensions
        },
        "dimension_counts": {k: by_dimension[k]["total"] for k in target_dimensions},
        "by_difficulty": {k: v["correct"] / v["total"] for k, v in by_difficulty.items() if v["total"]},
        "difficulty_counts": {k: v["total"] for k, v in by_difficulty.items()},
        "by_task_type": {k: v["correct"] / v["total"] for k, v in by_task_type.items() if v["total"]},
        "task_type_counts": {k: v["total"] for k, v in by_task_type.items()},
        "by_order": {k: v["correct"] / v["total"] for k, v in by_order.items() if v["total"]},
        "order_counts": {k: v["total"] for k, v in by_order.items()},
    }
