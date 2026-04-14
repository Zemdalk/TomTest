"""FollowBench metrics 计算

评测指标：
  - HSR (Hard Satisfaction Rate)：所有约束全部满足才为 1
  - SSR (Soft Satisfaction Rate)：满足约束数 / 总约束数
  - CSL (Consistent Satisfaction Levels)：从 L1 开始连续过关层数

评测方式：
  - 规则评测（Rule-based）：针对 13 个特定 source
  - LLM Judge 评测：其余 source，使用约束演进路径 prompt

输出结构：
  {
      "hsr_by_level": {1: float, ..., 5: float},
      "ssr_by_level": {1: float, ..., 5: float},
      "csl": float,
      "by_constraint_type": {
          "content": {"hsr_by_level": {...}, "ssr_by_level": {...}, "csl": float},
          ...
      },
      "total_groups": int,
      "llm_eval_count": int,
      "rule_eval_count": int,
  }
"""
from __future__ import annotations

import ast
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Rule-based evaluation helpers
# ---------------------------------------------------------------------------

RULE_BASED_SOURCES = {
    "E2E", "WIKIEVENTS", "CONLL2003", "text_editing",
    "cnn_dailymail", "xsum", "samsum", "gigaword", "arxiv",
    "BBH_logical", "BBH_time", "self_made_space", "gsm_8k",
}


def _contain_word(text: str, word: str) -> bool:
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r'[.!?]', text) if s.strip()])


def _count_sentence_words_less(text: str, n: int) -> bool:
    return all(len(s.split()) < n for s in re.split(r'[.!?]', text) if s.strip())


def _count_sentence_words_more(text: str, n: int) -> bool:
    return all(len(s.split()) > n for s in re.split(r'[.!?]', text) if s.strip())


def _n_sentence_contain_word(text: str, n: int, word: str) -> bool:
    sents = [s for s in re.split(r'[.!?]', text) if s.strip()]
    return len(sents) >= n and _contain_word(sents[n - 1], word)


def _n_sentence_is_present_perfect(text: str, n: int) -> bool:
    sents = [s for s in re.split(r'[.!?]', text) if s.strip()]
    return len(sents) >= n and bool(re.search(r'\b(has|have)\s+\w+', sents[n - 1]))


def _is_present_continuous(sentence: str) -> bool:
    return bool(re.search(r'\b(am|is|are)\s+\w+ing\b', sentence))


def _paragraphs_start_with_number_marker(text: str) -> bool:
    for para in text.split('\n\n'):
        para = para.lstrip()
        if not para:
            continue
        if not re.match(r'\d+\.', para):
            return False
    return True


def _rule_eval_format_22(generation: str, level: int) -> bool:
    paras = generation.split('\n\n')
    if level == 1:
        return len(paras) == 3
    elif level == 2:
        return len(paras) == 3 and all(_count_sentences(p) < 3 for p in paras)
    elif level == 3:
        return (len(paras) == 3
                and all(_count_sentences(p) < 3 for p in paras)
                and _count_sentence_words_more(generation, 20))
    elif level == 4:
        return (len(paras) == 3
                and all(_count_sentences(p) < 3 for p in paras)
                and _count_sentence_words_more(generation, 20)
                and _paragraphs_start_with_number_marker(generation))
    else:
        return (len(paras) == 4
                and all(_count_sentences(p) < 3 for p in paras[:3])
                and _count_sentence_words_more(generation, 20)
                and _paragraphs_start_with_number_marker(generation)
                and generation.endswith("Those are suggestions"))


def _rule_eval_format_30(generation: str, level: int) -> bool:
    c = [False] * 5
    c[0] = "**" in generation
    c[1] = len(re.findall(r'\*\*[\d].+?:', generation)) == 5
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generation)
    kw_sents = [s for s in sents if any(str(n) + '.' in s for n in range(1, 6))]
    c[2] = all(len(s.split('.')) < 3 for s in kw_sents)
    c[3] = all(10 <= len(s.split()) <= 15 for s in sents)
    c[4] = all(not w.endswith('-ly') for s in sents for w in s.split())
    return all(c[:level])


def _rule_eval(source: str, generation: str, target: str,
               level: int, example_id: int, category: str) -> Optional[Tuple[int, float]]:
    """
    Apply rule-based evaluation. Returns (hard, soft) or None if no rule applies.
    For level-aware rules all constraints at levels 1..level are checked together,
    so (hard, soft) is always (0 or 1, 0.0 or 1.0) for rule-based sources.
    """
    gen = generation.strip()

    if category == "format":
        if example_id == 22:
            ok = _rule_eval_format_22(gen, level)
        elif example_id == 30:
            ok = _rule_eval_format_30(gen, level)
        else:
            return None
        return (1, 1.0) if ok else (0, 0.0)

    if category == "example":
        if source not in RULE_BASED_SOURCES:
            return None
        pattern = target.replace('{{', '{').replace('}}', '}').replace('{answer}', '.*')
        pattern = re.escape(pattern).replace('\\.\\*', '.*')
        ok = bool(re.fullmatch(pattern, gen))
        return (1, 1.0) if ok else (0, 0.0)

    # Source-specific rules
    if source == "E2E":
        ok = gen == target
    elif source == "WIKIEVENTS":
        tgt_lines = target.strip().split('\n')
        ok = all(line in tgt_lines for line in gen.split('\n') if line.strip())
    elif source == "CONLL2003":
        try:
            ok = gen in ast.literal_eval(target)
        except Exception:
            ok = False
    elif source == "text_editing":
        ok = gen == target
    elif source == "cnn_dailymail":
        if level == 1:
            ok = _count_sentences(gen) == 3
        elif level == 2:
            ok = _count_sentences(gen) == 3 and _count_sentence_words_less(gen, 15)
        elif level == 3:
            ok = (_count_sentences(gen) == 3 and _count_sentence_words_less(gen, 15)
                  and _n_sentence_contain_word(gen, 1, 'Potter')
                  and _n_sentence_contain_word(gen, 2, 'actor')
                  and _n_sentence_contain_word(gen, 3, 'films'))
        elif level == 4:
            ok = (_count_sentences(gen) == 3 and _count_sentence_words_less(gen, 15)
                  and _n_sentence_contain_word(gen, 1, 'Potter')
                  and _n_sentence_contain_word(gen, 2, 'actor')
                  and _n_sentence_contain_word(gen, 3, 'films')
                  and not _n_sentence_contain_word(gen, 2, 'lavish'))
        else:
            ok = (_count_sentences(gen) == 3 and _count_sentence_words_less(gen, 15)
                  and _n_sentence_contain_word(gen, 1, 'Potter')
                  and _n_sentence_contain_word(gen, 2, 'actor')
                  and _n_sentence_contain_word(gen, 3, 'films')
                  and not _n_sentence_contain_word(gen, 2, 'lavish')
                  and _n_sentence_is_present_perfect(gen, 3))
        return (1, 1.0) if ok else (0, 0.0)
    elif source == "xsum":
        if level == 1:
            ok = _count_sentences(gen) == 1
        elif level == 2:
            ok = _count_sentences(gen) == 1 and len(gen.split()) < 20
        elif level == 3:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) < 20
                  and not _contain_word(gen, 'Newton Stewart'))
        elif level == 4:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) < 20
                  and not _contain_word(gen, 'Newton Stewart')
                  and _is_present_continuous(gen))
        else:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) < 20
                  and not _contain_word(gen, 'Newton Stewart')
                  and _is_present_continuous(gen)
                  and _contain_word(gen, 'operation'))
        return (1, 1.0) if ok else (0, 0.0)
    elif source == "samsum":
        if level == 1:
            ok = _count_sentences(gen) == 1
        elif level == 2:
            ok = _count_sentences(gen) == 1 and len(gen.split()) < 15
        elif level == 3:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) < 15
                  and _contain_word(gen, 'stuff'))
        elif level == 4:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) < 15
                  and _contain_word(gen, 'stuff')
                  and not bool(re.search(r'procrast', gen, re.IGNORECASE)))
        else:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) < 15
                  and _contain_word(gen, 'stuff')
                  and not bool(re.search(r'procrast', gen, re.IGNORECASE))
                  and sum(1 for c in gen if c in string.punctuation) == 1)
        return (1, 1.0) if ok else (0, 0.0)
    elif source == "gigaword":
        if level == 1:
            ok = len(gen.split()) == 8
        elif level == 2:
            ok = len(gen.split()) == 8 and not any(c in string.punctuation for c in gen)
        elif level == 3:
            ok = len(gen.split()) == 8 and not any(c in string.punctuation for c in gen) and gen.islower()
        elif level == 4:
            ok = (len(gen.split()) == 8 and not any(c in string.punctuation for c in gen)
                  and gen.islower() and not _contain_word(gen, 'bus'))
        else:
            ok = (len(gen.split()) == 8 and not any(c in string.punctuation for c in gen)
                  and gen.islower() and not _contain_word(gen, 'bus') and _contain_word(gen, 'in'))
        return (1, 1.0) if ok else (0, 0.0)
    elif source == "arxiv":
        if level == 1:
            ok = _count_sentences(gen) == 1
        elif level == 2:
            ok = _count_sentences(gen) == 1 and len(gen.split()) <= 20
        elif level == 3:
            ok = _count_sentences(gen) == 1 and len(gen.split()) <= 20 and gen[:2] == "We"
        elif level == 4:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) <= 20
                  and gen[:2] == "We" and _contain_word(gen, 'activations'))
        else:
            ok = (_count_sentences(gen) == 1 and len(gen.split()) <= 20
                  and gen[:2] == "We" and _contain_word(gen, 'activations')
                  and not _contain_word(gen, 'transformer'))
        return (1, 1.0) if ok else (0, 0.0)
    elif source == "BBH_logical":
        match = re.findall(r'\(([A-Z])\)', gen)
        ok = match[-1] == target if match else False
    elif source == "BBH_time":
        match = re.findall(r'\d{2}/\d{2}/\d{4}', gen)
        ok = match[-1] == target if match else False
    elif source == "self_made_space":
        ok = target in gen
    elif source == "gsm_8k":
        match = re.findall(r'\$\d+', gen)
        ok = match[-1] == target if match else False
    else:
        return None

    return (1, 1.0) if ok else (0, 0.0)


# ---------------------------------------------------------------------------
# LLM judge prompt builder + response parser
# ---------------------------------------------------------------------------

def _build_judge_prompt(constraint_type: str, evolve_instructions: List[str], answer: str) -> str:
    """Build the FollowBench-style LLM judge prompt (constraint evolution path)."""
    level = len(evolve_instructions) - 1  # evolve_instructions[0] = L0 base
    type_desc = {
        "content":  "content constraint",
        "situation": "situation constraint",
        "style":    "style constraint",
        "format":   "format constraint",
        "mixed":    "constraint",
    }.get(constraint_type, "constraint")

    if level == 1:
        return (
            f"Given an initial instruction, we add one {type_desc} and obtain the final instruction "
            f"with 1 additional constraint.\n\n"
            f"#Initial Instruction#\n{evolve_instructions[0]}\n\n"
            f"#Initial Instruction + 1 constraint#\n{evolve_instructions[1]}\n\n"
            f"#Answer of Initial Instruction + 1 constraint#\n{answer}\n\n"
            f"#System#\n"
            f"1) Please identify the 1 added constraint.\n"
            f"2) Please discriminate if the answer satisfies the 1 added constraint.\n"
            f"3) In the final line, only output a Python LIST with 1 element ('YES' or 'NO')."
        )
    else:
        prompt = (
            f"Given an initial instruction, we add one {type_desc} per time and obtain the final "
            f"instruction with {level} additional constraints.\n\n"
            f"#Initial Instruction#\n{evolve_instructions[0]}\n\n"
            f"#Initial Instruction + 1 constraint#\n{evolve_instructions[1]}\n\n"
        )
        for i in range(2, level + 1):
            prompt += f"#Initial Instruction + {i} constraints#\n{evolve_instructions[i]}\n\n"
        prompt += (
            f"#Answer of Initial Instruction + {level} constraints#\n{answer}\n\n"
            f"#System#\n"
            f"1) Please identify all {level} added constraints.\n"
            f"2) For each constraint, judge if the answer satisfies it.\n"
            f"3) In the final line, only output a Python LIST with {level} elements ('YES' or 'NO')."
        )
        return prompt


def _parse_judge_response(response: str, level: int) -> Tuple[int, float]:
    """Parse LLM judge response. Returns (hard_satisfy, soft_satisfy) or (-1, -1.0) on failure."""
    try:
        last_line = response.strip().rstrip('`').split('\n')[-1]
        if level == 1:
            if 'YES' in last_line:
                return 1, 1.0
            elif 'NO' in last_line:
                return 0, 0.0
            raise ValueError("no YES/NO found")
        else:
            match = re.search(r'\[.*\]', last_line)
            if match:
                sat_list = ast.literal_eval(match.group())
                if len(sat_list) == level:
                    n_yes = sum(1 for x in sat_list if x == 'YES')
                    return (1 if n_yes == level else 0), n_yes / level
            raise ValueError("cannot parse list")
    except Exception:
        return -1, -1.0


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[str],
    data: List[Dict[str, Any]],
    evolution_paths: Dict[Tuple, Dict[int, str]],
    judge_client=None,
) -> Dict[str, Any]:
    """Compute FollowBench metrics: HSR, SSR, CSL (overall + per constraint type).

    Args:
        predictions:      Model responses, aligned with data (level 1-5 items only).
        data:             Data rows (level 1-5 only), each with Meta fields.
        evolution_paths:  {(constraint_type, example_group_id): {level: instruction}}
        judge_client:     LLMClient for LLM judge. If None, LLM eval items are skipped.

    Returns:
        Metrics dict with HSR/SSR/CSL overall and by constraint_type.
    """
    # Accumulate per-group results: {(ctype, group_id): {level: (hard, soft)}}
    group_results: Dict[Tuple, Dict[int, Tuple[int, float]]] = defaultdict(dict)

    # Batch judge prompts to send (deferred for batching)
    judge_queue: List[Tuple] = []  # (prompt, ctype, group_id, level)

    rule_count = 0
    llm_count = 0

    for row, response in zip(data, predictions):
        meta = row["Meta"]
        ctype = meta["constraint_type"]
        group_id = meta["example_group_id"]
        level = meta["constraint_level"]
        source = meta["source_dataset"]
        target = (row["Answer"]["Correct_Answer"] or [""])[0]

        # Try rule-based first
        rule_result = _rule_eval(source, response, target, level, group_id, ctype)
        if rule_result is not None:
            group_results[(ctype, group_id)][level] = rule_result
            rule_count += 1
            continue

        # Queue for LLM judge
        epath = evolution_paths.get((ctype, group_id), {})
        instrs = [epath.get(i, "") for i in range(0, level + 1)]
        if judge_client is not None and all(instrs):
            judge_prompt = _build_judge_prompt(ctype, instrs, response)
            judge_queue.append((judge_prompt, ctype, group_id, level))
        else:
            # No judge available: mark as skipped (-1)
            group_results[(ctype, group_id)][level] = (-1, -1.0)

    # Run LLM judge in batch
    if judge_queue and judge_client is not None:
        prompts = [item[0] for item in judge_queue]
        from FollowBench.schemas import JudgeAnswer  # noqa: local import

        # Use free-text generate since judge response needs parsing (not structured)
        judge_responses = judge_client.batch_generate(prompts)
        for (_, ctype, group_id, level), resp in zip(judge_queue, judge_responses):
            hard, soft = _parse_judge_response(resp, level)
            group_results[(ctype, group_id)][level] = (hard, soft)
            llm_count += 1

    # Aggregate HSR / SSR / CSL
    def aggregate(groups: Dict[Tuple, Dict[int, Tuple[int, float]]]) -> Dict[str, Any]:
        hsr_by_level: Dict[int, List[int]] = defaultdict(list)
        ssr_by_level: Dict[int, List[float]] = defaultdict(list)
        csl_values: List[float] = []

        for (_, _gid), level_results in groups.items():
            # CSL: consecutive levels from L1
            csl = 0
            for lv in range(1, 6):
                hr, _ = level_results.get(lv, (-1, -1.0))
                if hr == 1:
                    csl += 1
                else:
                    break
            csl_values.append(float(csl))

            for lv, (hr, sr) in level_results.items():
                if hr == -1:  # skipped
                    continue
                hsr_by_level[lv].append(hr)
                ssr_by_level[lv].append(sr)

        result = {}
        for lv in range(1, 6):
            h_list = hsr_by_level.get(lv, [])
            s_list = ssr_by_level.get(lv, [])
            result[f"hsr_L{lv}"] = sum(h_list) / len(h_list) if h_list else None
            result[f"ssr_L{lv}"] = sum(s_list) / len(s_list) if s_list else None

        result["hsr_by_level"] = {lv: result.pop(f"hsr_L{lv}") for lv in range(1, 6)}
        result["ssr_by_level"] = {lv: result.pop(f"ssr_L{lv}") for lv in range(1, 6)}
        result["csl"] = sum(csl_values) / len(csl_values) if csl_values else 0.0
        result["n_groups"] = len(groups)
        return result

    # Overall
    overall = aggregate(group_results)

    # Per constraint type
    by_type: Dict[str, Dict] = {}
    for ctype in ["content", "situation", "style", "format", "example", "mixed"]:
        type_groups = {k: v for k, v in group_results.items() if k[0] == ctype}
        if type_groups:
            by_type[ctype] = aggregate(type_groups)

    # Primary scalar metric: mean HSR across L1-L5 (for framework compatibility)
    valid_hsr = [v for v in overall["hsr_by_level"].values() if v is not None]
    mean_hsr = sum(valid_hsr) / len(valid_hsr) if valid_hsr else 0.0

    return {
        # Primary metric (for runner compatibility)
        "accuracy": mean_hsr, # mean hsr as primary accuracy metric
        "correct": int(mean_hsr * overall["n_groups"]),
        "total": overall["n_groups"],
        # FollowBench metrics
        "hsr_by_level": overall["hsr_by_level"],
        "ssr_by_level": overall["ssr_by_level"],
        "csl": overall["csl"],
        "n_groups": overall["n_groups"],
        "by_constraint_type": by_type,
        "rule_eval_count": rule_count,
        "llm_eval_count": llm_count,
    }
