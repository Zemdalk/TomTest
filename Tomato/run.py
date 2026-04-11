"""Tomato 评测脚本（结构化输出 MCQAnswer，与 ToMBench 推理路径一致）"""
from __future__ import annotations

import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Match, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from Tomato.prompts import get_template, build_prompt
from Tomato.metrics import compute_metrics
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 从原始样本解析 MCQ，写入 row["_mcq"]（供 build_prompt 与金标）
# ---------------------------------------------------------------------------

CHOICE_LETTERS = tuple("ABCD")
_pat_options = re.compile(r"\b([A-D])\s*[\.．、\):：]\s*", flags=re.IGNORECASE)


def to_json_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def extract_answers(answer_obj: Any) -> List[str]:
    if isinstance(answer_obj, dict):
        cands = answer_obj.get("Correct_Answer") or answer_obj.get("Correct Answer", [])
        if isinstance(cands, list):
            return [str(x) for x in cands if str(x).strip()]
        if isinstance(cands, str) and cands.strip():
            return [cands]
    if isinstance(answer_obj, list):
        return [str(x) for x in answer_obj if str(x).strip()]
    if isinstance(answer_obj, str) and answer_obj.strip():
        return [answer_obj]
    return []


def extract_wrong_answers(answer_obj: Any) -> List[str]:
    if isinstance(answer_obj, dict):
        w = answer_obj.get("Wrong_Answer") or answer_obj.get("Wrong Answer", [])
        if isinstance(w, list):
            return [str(x) for x in w if str(x).strip()]
        if isinstance(w, str) and w.strip():
            return [w]
    return []


def normalize_tom_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if isinstance(out.get("Story"), str):
        out["Story"] = {"full_story": out["Story"].strip(), "summary": "", "background": []}
    if isinstance(out.get("State"), dict):
        st = dict(out["State"])
        if "Human State" in st and "Human_State" not in st:
            st["Human_State"] = st.pop("Human State")
        if "Environment State" in st and "Environment_State" not in st:
            st["Environment_State"] = st.pop("Environment State")
        out["State"] = st
    if isinstance(out.get("Answer"), dict):
        ans = dict(out["Answer"])
        if "Correct_Answer" not in ans and "Correct Answer" in ans:
            ans["Correct_Answer"] = ans["Correct Answer"]
        if "Wrong_Answer" not in ans and "Wrong Answer" in ans:
            ans["Wrong_Answer"] = ans["Wrong Answer"]
        out["Answer"] = ans
    return out


def _normalize_choice_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\r\n\"'`.,;:!?()[]{}")


def canonicalize_choice_letter(text: str, option_letters: Sequence[str]) -> str:
    if not text:
        return ""
    allowed = "".join(option_letters)
    upper = text.strip().upper()
    if upper in option_letters:
        return upper
    m = re.search(
        rf"^\s*[\[\(\{{<]?([{allowed}])[\]\)\}}>]?\s*[\.\)\]:：、]?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    return ""


def resolve_gold_letter(
    correct_text: str,
    option_letters: Sequence[str],
    option_texts: Sequence[str],
) -> str:
    gold = canonicalize_choice_letter(correct_text, option_letters)
    if gold in option_letters:
        return gold
    norm_correct = _normalize_choice_text(correct_text)
    if not norm_correct:
        return ""
    for letter, option_text in zip(option_letters, option_texts):
        if norm_correct == _normalize_choice_text(option_text):
            return letter
    return ""


def _extract_story_text(story: Any) -> str:
    if story is None:
        return ""
    if isinstance(story, dict):
        parts = [str(story["full_story"])] if story.get("full_story") else []
        if story.get("summary"):
            parts.append(f"Summary: {story['summary']}")
        if story.get("background"):
            parts.append(f"Background: {to_json_text(story['background'])}")
        return "\n".join(parts).strip()
    return str(story)


def _first_after(matches: List[Match[str]], letter: str, start_pos: int) -> Optional[Match[str]]:
    for m in matches:
        if m.start() > start_pos and m.group(1).upper() == letter:
            return m
    return None


def _task_type_from_row(row: Dict[str, Any]) -> str:
    meta = row.get("Meta")
    if isinstance(meta, dict):
        for k in ("task_type", "Task_Type", "task", "dimension"):
            v = meta.get(k)
            if v:
                return str(v)
    return "False Belief"


def _first_nonempty(row: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_explicit_letter_options(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
        text = _first_nonempty(row, candidates)
        if not text:
            break
        text = re.sub(rf"^\s*{letter}\s*[\.．、\):：]\s*", "", text, flags=re.IGNORECASE)
        options[letter] = text.strip()
    if len(options) < 2:
        return None
    question = _first_nonempty(row, ["Question", "QUESTION", "question", "问题"])
    return {
        "question": question,
        "option_letters": list(options.keys()),
        "option_texts": [options[k] for k in options.keys()],
        "options": options,
    }


def extract_unshuffled_mcq(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    row = normalize_tom_row(dict(row))
    correct_list = extract_answers(row.get("Answer"))
    wrong_list = extract_wrong_answers(row.get("Answer"))

    explicit = _extract_explicit_letter_options(row)
    if explicit is not None:
        option_letters = explicit["option_letters"]
        option_texts = explicit["option_texts"]
        gold_raw = resolve_gold_letter(
            (correct_list[0] if correct_list else _first_nonempty(row, ["答案", "answer", "ANSWER"])),
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
                "task_type": _task_type_from_row(row),
            }

    if len(correct_list) == 1 and 1 <= len(wrong_list) <= 3:
        option_letters = list(CHOICE_LETTERS[: 1 + len(wrong_list)])
        opts = [correct_list[0]] + wrong_list[: len(option_letters) - 1]
        original_choices = {option_letters[i]: str(opts[i]).strip() for i in range(len(option_letters))}
        gold_raw = option_letters[0]
        story = _extract_story_text(row.get("Story"))
        question = str(row.get("Question", "")).strip()
        return {
            "story": story,
            "question": question,
            "original_choices": original_choices,
            "gold_letter": gold_raw,
            "num_options": len(option_letters),
            "task_type": _task_type_from_row(row),
        }

    q = str(row.get("Question", ""))
    ms = list(_pat_options.finditer(q))
    for a_match in ms:
        if a_match.group(1).upper() != "A":
            continue
        chosen = [a_match]
        prev = a_match.start()
        for letter in CHOICE_LETTERS[1:]:
            nxt = _first_after(ms, letter, prev)
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
            "task_type": _task_type_from_row(row),
        }

    return None


def preprocess_mcq(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """预处理：解析 MCQ 并将结果写入 row['_mcq']，过滤无效行。"""
    valid: List[Dict[str, Any]] = []
    for row in data:
        parsed = extract_unshuffled_mcq(row)
        if parsed is not None:
            row["_mcq"] = parsed
            valid.append(row)
    if not valid:
        raise RuntimeError("没有可评测的 MCQ 样本（extract_unshuffled_mcq 全为 None）。")
    return valid


def shuffle_mcq_options(mcq: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """返回一份选项顺序被打乱的 _mcq 副本，gold_letter 同步更新。"""
    rng = random.Random(seed)
    letters = sorted(mcq["original_choices"].keys())
    texts = [mcq["original_choices"][l] for l in letters]
    old_gold_idx = letters.index(mcq["gold_letter"])

    indices = list(range(len(letters)))
    rng.shuffle(indices)

    new_choices: Dict[str, str] = {}
    new_gold = mcq["gold_letter"]
    for new_pos, old_idx in enumerate(indices):
        new_choices[letters[new_pos]] = texts[old_idx]
        if old_idx == old_gold_idx:
            new_gold = letters[new_pos]

    return {**mcq, "original_choices": new_choices, "gold_letter": new_gold}


def _answer_from_structured(r: Any) -> str:
    """从 batch_generate_structure 返回的 Pydantic 对象取出选项字母。"""
    a = getattr(r, "answer", None)
    return str(a) if a is not None else ""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> None:
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("Tomato/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]

    # 获取 prompt 模板
    template = get_template(prompt_method)

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"])

    # 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["subset"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} raw rows from {dataset_config['subset']}")

    # Tomato 特有：MCQ 预处理与过滤（将解析结果写入 row['_mcq']）
    data = preprocess_mcq(data)

    repeats = experiment_config["repeats"]
    print(f"MCQ samples: {len(data)}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {repeats} (each with different option shuffle)")

    # 每个 repeat 使用不同 seed 打乱选项顺序，消除位置偏差
    all_prompts: List[str] = []
    repeat_data: List[List[Dict[str, Any]]] = []

    for i in range(repeats):
        shuffled_rows: List[Dict[str, Any]] = []
        for j, row in enumerate(data):
            shuffled_mcq = shuffle_mcq_options(row["_mcq"], seed=42 * (i + 1) + j)
            shuffled_row = dict(row)
            shuffled_row["_mcq"] = shuffled_mcq
            shuffled_rows.append(shuffled_row)
            all_prompts.append(build_prompt(template, shuffled_row))
        repeat_data.append(shuffled_rows)

    # 批量结构化推理（与 ToMBench 一致）
    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    # 使用数据集的 metrics 函数计算
    n = len(data)
    all_predictions: List[List[str]] = []
    all_metrics: List[Dict[str, Any]] = []
    all_gold: List[List[str]] = []

    for i in range(repeats):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        rows = repeat_data[i]
        predictions = [_answer_from_structured(r) for r in repeat_results]
        all_predictions.append(predictions)

        # 调用数据集的 metrics 函数
        metrics = compute_metrics(predictions, rows)
        all_metrics.append(metrics)
        all_gold.append([row["_mcq"]["gold_letter"] for row in rows])
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    # 保存结果（per-repeat gold，因为每轮 shuffle 不同）
    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=all_gold,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, repeats, n)


if __name__ == "__main__":
    main()
