"""答案打分：模式推断 + 多种匹配策略。"""
import re
from typing import Any, Dict, List, Optional, Tuple


DATASET_SCORING: Dict[str, str] = {
    "ToMBench": "mcq_letter",
    "ToMi":     "open_substring",
    "OpenToM":  "open_substring",
    "BigToM":   "open_substring",
}
_SCORING_MODES = frozenset({"auto", "mcq_letter", "open_substring", "open_normalized", "yes_no"})


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split()).strip(".,!?;:\"'` ")


def _is_mcq_letter(s: str) -> bool:
    return bool(s and len(s) == 1 and s.lower() in "abcdefghij")


def _last_option_letter(text: str) -> Optional[str]:
    found = re.findall(r"\b([a-j])\b", text, flags=re.IGNORECASE)
    skip = {"i", "a"}
    for x in reversed(found):
        if x.lower() not in skip:
            return x.lower()
    return found[-1].lower() if found else None


def extract_mcq_letter(pred_n: str) -> Optional[str]:
    """从归一化模型输出中提取最终选项字母（取最靠右的匹配）。"""
    if not pred_n:
        return None
    patterns = [
        r"\banswer\s+is\s+\(?([a-j])\)?",
        r"(?:^|[\s.!?\n])(?:the\s+)?(?:final\s+)?answer\s+is\s+\(?([a-j])\)?",
        r"(?:^|[\s.!?\n])answer\s*[:：]\s*\(?([a-j])\)?",
        r"(?:option|choice)\s*[:：]\s*\(?([a-j])\)?",
        r"\boption\s+([a-j])\b",
        r"(?:答案|选项|选择|正确答案)\s*[:：]\s*\(?([a-j])\)?",
        r"\(([a-j])\)\s*(?:is\s+)?(?:correct|right|对)\b",
        r"(?:^|[\s\"'「])([a-j])\s*[).、．]\s*(?:$|[\s\n])",
        r"\*\*([a-j])\*\*",
        r"(?:^|[\s\n])`([a-j])`(?:\s|$)",
    ]
    last, last_start = None, -1
    for p in patterns:
        for m in re.finditer(p, pred_n, flags=re.IGNORECASE):
            if m.start() >= last_start:
                last_start, last = m.start(), m.group(1).lower()
    if last is not None:
        return last
    # 长 CoT：优先取末段独立字母
    if len(pred_n) > 400:
        t = _last_option_letter(pred_n[-1200:])
        if t is not None:
            return t
    return _last_option_letter(pred_n)


def infer_scoring_mode(dataset: str, row: Dict[str, Any]) -> str:
    """按优先级推断打分模式：Meta 字段 > 数据集名表 > auto。"""
    meta = row.get("Meta")
    if isinstance(meta, dict):
        raw = meta.get("scoring_mode") or meta.get("eval_mode")
        if isinstance(raw, str):
            key = raw.strip().lower().replace("-", "_")
            if key in _SCORING_MODES:
                return key
        tt = str(meta.get("task_type", "") or meta.get("question_type", "") or "").lower()
        if tt in ("multiple_choice", "mcq", "multi_choice", "choice"):
            return "mcq_letter"
        if tt in ("open", "qa", "free_form", "freeform", "generation"):
            return "open_substring"
        if tt in ("yes_no", "yesno", "binary"):
            return "yes_no"
    return DATASET_SCORING.get(dataset, "auto")


def score_against_gold(pred_n: str, gold_n: str, mode: str) -> bool:
    if not gold_n:
        return False
    if mode == "mcq_letter":
        if _is_mcq_letter(gold_n):
            got = extract_mcq_letter(pred_n)
            return got is not None and got == gold_n.lower()
        return gold_n in pred_n or pred_n in gold_n
    if mode == "open_substring":
        return bool(gold_n in pred_n or pred_n in gold_n)
    if mode == "open_normalized":
        return pred_n == gold_n
    if mode == "yes_no":
        def canon(s: str) -> Optional[str]:
            t = s.strip().lower()
            if re.fullmatch(r"(yes|y|true|是|对|正确)", t): return "yes"
            if re.fullmatch(r"(no|n|false|否|不|错误)", t):  return "no"
            return None
        g = canon(gold_n)
        if g is None:
            return False
        low = pred_n.lower()
        yes_m = re.search(r"\b(yes|y|true|是|对|正确)\b", low)
        no_m  = re.search(r"\b(no|n|false|否|不|错误)\b", low)
        if yes_m and (not no_m or yes_m.start() < no_m.start()):
            p = "yes"
        elif no_m:
            p = "no"
        else:
            p = canon(pred_n)
        return p is not None and p == g
    return False


def score_prediction(
    pred_n: str, answers_n: List[str], dataset: str, row: Dict[str, Any]
) -> Tuple[bool, str]:
    mode = infer_scoring_mode(dataset, row)
    if mode == "auto":
        non_empty = [a for a in answers_n if a]
        mode = "mcq_letter" if non_empty and all(_is_mcq_letter(a) for a in non_empty) else "open_substring"
    for a in answers_n:
        if a and score_against_gold(pred_n, a, mode):
            return True, mode
    return False, mode
