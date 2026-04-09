"""答案打分：模式推断 + 多种匹配策略。"""
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data import extract_answers, extract_wrong_answers, normalize_tom_row
from prompt import CHOICE_LETTERS, _extract_story_text, resolve_gold_letter


DATASET_SCORING: Dict[str, str] = {
    "ToMBench": "mcq_letter",
    "SocialIQA": "mcq_letter",
    "ToMi":     "open_substring",
    "OpenToM":  "open_substring",
    "BigToM":   "open_substring",
}
_SCORING_MODES = frozenset({"auto", "mcq_letter", "open_substring", "open_normalized", "yes_no"})


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split()).strip(".,!?;:\"'` ")


def _is_mcq_letter(s: str) -> bool:
    return bool(s and len(s) == 1 and s.lower() in "abcdefghij")


def _allowed_letters_regex(option_letters: Optional[Iterable[str]] = None) -> str:
    letters = "".join((x.lower() for x in (option_letters or tuple("ABCDEFGHIJ"))))
    return re.escape(letters)


def _option_letters_char_class(option_letters: Optional[Iterable[str]] = None) -> str:
    """用于 [...] 字符类，仅含 A–J 单字母，无需转义。"""
    letters = [x.upper() if isinstance(x, str) else str(x).upper() for x in (option_letters or tuple("ABCDEFGHIJ"))]
    uniq = "".join(sorted({c[0].lower() for c in letters if c}))
    return uniq


def strip_thinking_blocks(text: str) -> str:
    """去掉常见思考块，避免干扰 <answer> 抽取（与评测脚本行为对齐）。"""
    t = text or ""
    for pat, flags in (
        (r"<redacted_thinking>.*?</redacted_thinking>", re.DOTALL | re.IGNORECASE),
    ):
        t = re.sub(pat, "", t, flags=flags)
    return t


def extract_answer_tag_content(text: str) -> str:
    """优先返回一对 <answer>...</answer> 内文本；否则未闭合 <answer> 至文末；否则原文本 strip 后。"""
    t = text or ""
    m = re.search(r"<answer>(.*?)</answer>", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    m2 = re.search(r"<answer>(.*?)$", t, flags=re.DOTALL | re.IGNORECASE)
    if m2:
        return (m2.group(1) or "").strip()
    return t.strip()


def prepare_mcq_prediction_for_extraction(raw: str) -> str:
    return extract_answer_tag_content(strip_thinking_blocks(raw))


def normalize_mcq_choice_letter(answer: str, option_letters: Optional[Iterable[str]] = None) -> Optional[str]:
    """从短字符串中解析选项字母（参考用户脚本，选项集由 option_letters 限定）。"""
    if not (answer or "").strip():
        return None
    chars = _option_letters_char_class(option_letters)
    if not chars:
        return None
    s = answer.strip()
    m1 = re.match(rf"^([{chars}])[.\、\)\s]*", s, re.IGNORECASE)
    if m1:
        return m1.group(1).lower()
    m2 = re.match(rf"^([{chars}])[.\s]+", s, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    m3 = re.search(r"(?:answer|选|选择|option|is)\s*([{chars}])", s, re.IGNORECASE)
    if m3:
        return m3.group(1).lower()
    return None


def _last_option_letter(text: str, option_letters: Optional[Iterable[str]] = None) -> Optional[str]:
    allowed = _allowed_letters_regex(option_letters)
    found = re.findall(rf"\b([{allowed}])\b", text, flags=re.IGNORECASE)
    skip = {"i", "a"}
    for x in reversed(found):
        if len(found) == 1 or x.lower() not in skip:
            return x.lower()
    return found[-1].lower() if found else None


def extract_mcq_letter(pred: str, option_letters: Optional[Iterable[str]] = None) -> Optional[str]:
    """从模型输出中提取 MCQ 选项字母：先去思考块、再取 <answer>…、再做字母规范化，最后走原有正则兜底。"""
    prepared = prepare_mcq_prediction_for_extraction(pred or "")
    letter = normalize_mcq_choice_letter(prepared, option_letters)
    if letter is not None:
        return letter
    pred_n = normalize_text(prepared)
    if not pred_n:
        return None
    allowed = _allowed_letters_regex(option_letters)
    patterns = [
        rf"\banswer\s+is\s+\(?([{allowed}])\)?",
        rf"(?:^|[\s.!?\n])(?:the\s+)?(?:final\s+)?answer\s+is\s+\(?([{allowed}])\)?",
        rf"(?:^|[\s.!?\n])answer\s*[:：]\s*\(?([{allowed}])\)?",
        rf"(?:option|choice)\s*[:：]\s*\(?([{allowed}])\)?",
        rf"\boption\s+([{allowed}])\b",
        rf"(?:答案|选项|选择|正确答案)\s*[:：]\s*\(?([{allowed}])\)?",
        rf"\(([{allowed}])\)\s*(?:is\s+)?(?:correct|right|对)\b",
        rf"\[\s*([{allowed}])\s*\]",
        rf"(?:^|[\s\"'「])([{allowed}])\s*[).、．]\s*(?:$|[\s\n])",
        rf"\*\*([{allowed}])\*\*",
        rf"(?:^|[\s\n])`([{allowed}])`(?:\s|$)",
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
        t = _last_option_letter(pred_n[-1200:], option_letters)
        if t is not None:
            return t
    return _last_option_letter(pred_n, option_letters)


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


def score_against_gold(pred_raw: str, pred_n: str, gold_n: str, mode: str) -> bool:
    if not gold_n:
        return False
    if mode == "mcq_letter":
        if _is_mcq_letter(gold_n):
            got = extract_mcq_letter(pred_raw)
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
    pred_raw: str, answers_n: List[str], dataset: str, row: Dict[str, Any]
) -> Tuple[bool, str]:
    pred_n = normalize_text(pred_raw)
    mode = infer_scoring_mode(dataset, row)
    if mode == "auto":
        non_empty = [a for a in answers_n if a]
        mode = "mcq_letter" if non_empty and all(_is_mcq_letter(a) for a in non_empty) else "open_substring"
    for a in answers_n:
        if a and score_against_gold(pred_raw, pred_n, a, mode):
            return True, mode
    return False, mode


# ---------------------------------------------------------------------------
# v2：MCQ 行解析（ToMBench / Tomato 共用）与模型输出解析
# ---------------------------------------------------------------------------

_pat_options_v2 = re.compile(r"\b([A-D])\s*[\.．、\):：]\s*", flags=re.IGNORECASE)


def _first_after_v2(matches: List[re.Match], letter: str, start_pos: int) -> Optional[re.Match]:
    for m in matches:
        if m.start() > start_pos and m.group(1).upper() == letter:
            return m
    return None


def _task_type_from_row_v2(row: Dict[str, Any]) -> str:
    meta = row.get("Meta")
    if isinstance(meta, dict):
        for k in ("task_type", "Task_Type", "task", "dimension"):
            v = meta.get(k)
            if v:
                return str(v)
    return "False Belief"


def _first_nonempty_v2(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_explicit_letter_options_v2(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    letters = list(CHOICE_LETTERS)
    options: Dict[str, str] = {}
    for letter in letters:
        candidates = [
            f"OPTION-{letter}",
            f"OPTION_{letter}",
            f"option-{letter}",
            f"option_{letter}",
            f"choice_{letter}",
            f"choice-{letter}",
            f"选项{letter}",
            letter,
        ]
        text = _first_nonempty_v2(row, candidates)
        if not text:
            break
        text = re.sub(rf"^\s*{letter}\s*[\.．、\):：]\s*", "", text, flags=re.IGNORECASE)
        options[letter] = text.strip()
    if len(options) < 2:
        return None
    question = _first_nonempty_v2(row, ["Question", "QUESTION", "question", "问题"])
    return {
        "question": question,
        "option_letters": list(options.keys()),
        "option_texts": [options[k] for k in options.keys()],
        "options": options,
    }


def extract_unshuffled_mcq(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    返回:
      story, question (展示用题干), original_choices: Dict[str,str], gold_letter, num_options, task_type
    无法解析为 MCQ 时返回 None。
    """
    row = normalize_tom_row(dict(row))
    correct_list = extract_answers(row.get("Answer"))
    wrong_list = extract_wrong_answers(row.get("Answer"))

    explicit = _extract_explicit_letter_options_v2(row)
    if explicit is not None:
        option_letters = explicit["option_letters"]
        option_texts = explicit["option_texts"]
        gold_raw = resolve_gold_letter(
            (correct_list[0] if correct_list else _first_nonempty_v2(row, ["答案", "answer", "ANSWER"])),
            option_letters,
            option_texts,
        )
        if gold_raw in option_letters:
            return {
                "story": _extract_story_text(row.get("Story")),
                "question": explicit["question"],
                "original_choices": explicit["options"],
                "gold_letter": gold_raw,
                "num_options": len(option_letters),
                "task_type": _task_type_from_row_v2(row),
            }

    if len(correct_list) == 1 and 1 <= len(wrong_list) <= 3:
        option_letters = list(CHOICE_LETTERS[: 1 + len(wrong_list)])
        opts = [correct_list[0]] + wrong_list[: len(option_letters) - 1]
        original_choices = {option_letters[i]: str(opts[i]).strip() for i in range(len(option_letters))}
        gold_raw = option_letters[0]
        if gold_raw not in option_letters:
            return None
        story = _extract_story_text(row.get("Story"))
        question = str(row.get("Question", "")).strip()
        return {
            "story": story,
            "question": question,
            "original_choices": original_choices,
            "gold_letter": gold_raw,
            "num_options": len(option_letters),
            "task_type": _task_type_from_row_v2(row),
        }

    q = str(row.get("Question", ""))
    ms = list(_pat_options_v2.finditer(q))
    for a_match in ms:
        if a_match.group(1).upper() != "A":
            continue
        chosen = [a_match]
        prev = a_match.start()
        for letter in CHOICE_LETTERS[1:]:
            nxt = _first_after_v2(ms, letter, prev)
            if nxt is None:
                break
            chosen.append(nxt)
            prev = nxt.start()
        if len(chosen) < 2:
            continue
        option_letters = list(CHOICE_LETTERS[: len(chosen)])
        opts = [
            q[chosen[j].end() : (chosen[j + 1].start() if j < len(chosen) - 1 else len(q))].strip()
            for j in range(len(chosen))
        ]
        while opts and not opts[-1]:
            opts.pop()
            option_letters.pop()
        if len(opts) < 2 or not all(opts):
            continue
        gold_raw = resolve_gold_letter(
            correct_list[0] if correct_list else "",
            option_letters,
            opts,
        )
        if gold_raw not in option_letters:
            continue
        story = _extract_story_text(row.get("Story"))
        question_stem = q[: chosen[0].start()].strip()
        original_choices = {option_letters[i]: opts[i] for i in range(len(option_letters))}
        return {
            "story": story,
            "question": question_stem,
            "original_choices": original_choices,
            "gold_letter": gold_raw,
            "num_options": len(option_letters),
            "task_type": _task_type_from_row_v2(row),
        }

    return None


def create_position_map(shuffled_choices: List[str], original_choices: Dict[str, str]) -> Dict[str, str]:
    """新位置标签 -> 原始选项字母（与用户提供逻辑一致）。"""
    original_map = {v: k for k, v in original_choices.items()}
    position_map: Dict[str, str] = {}
    for new_idx, choice_content in enumerate(shuffled_choices):
        new_label = chr(ord("A") + new_idx)
        if choice_content in original_map:
            position_map[new_label] = original_map[choice_content]
        else:
            position_map[new_label] = new_label
    return position_map


def extract_bracket_answer(text: str) -> Optional[str]:
    """ToMBench v2：从输出中解析 [[A]] 等形式。"""
    if not text:
        return None
    m = re.search(r"\[\s*([A-Da-d])\s*\]", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(\s*([A-Da-d])\s*\)", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:^|\s)([A-Da-d])[\.。)]", text)
    if m:
        return m.group(1).upper()
    return None


def extract_tomato_choice(text: str, allowed_letters: str) -> Optional[str]:
    """从输出中解析一个选项字母，allowed_letters 如 \"ABCD\" 或 \"AB\"。"""
    if not text:
        return None
    allowed = {c.upper() for c in allowed_letters}
    m = re.search(r"\[\s*([A-Da-d])\s*\]", text)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    m = re.search(r"\[\[?\s*([A-Da-d])\s*\]?\]", text)
    if m and m.group(1).upper() in allowed:
        return m.group(1).upper()
    for ch in allowed:
        if re.search(rf"\b{ch}\b", text.upper()):
            return ch
    return None
