"""Prompt 构建：模板加载、字段填充、MCQ 选项打乱。"""
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data import SampleMeta, extract_answers, extract_wrong_answers, to_json_text


# ---------------------------------------------------------------------------
# 嵌入式默认 prompt（prompt/*.txt 存在时被覆盖）
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE = """You are a helpful reasoning assistant.
Read the sample and answer the question briefly.

[Story]
{story}

[Action]
{action}

[State]
{state}

[Meta]
{meta}

{options_block}

[Question]
{question}

Return only the final answer text.
"""

EMBEDDED_MAIN_MCQ_ABCD = """You are given structured ToM task inputs.

Fields:
- Story: narrative context
- Actions: observed action sequence
- Human State: current human-related state
- Environment State: current environment-related state
- Question: target question to answer

Use only the provided fields as input context.

Output requirements:
1) Return exactly one final answer.
2) Do not output reasoning, explanation, or any extra text.
3) Follow this exact format:
Output your final verdict by strictly following this format: [A], [B], [C], or [D]"""

EMBEDDED_MAIN_OPEN = """You are given structured ToM task inputs.

Fields:
- Story: narrative context
- Actions: observed action sequence
- Human State: current human-related state
- Environment State: current environment-related state
- Question: target question to answer

Use only the provided fields as input context.

Output requirements:
1) Return exactly one short final answer.
2) Do not output reasoning, explanation, or any extra text.
3) Reply with the answer text only (no brackets unless the answer itself requires them)."""

EMBEDDED_MAIN2_MCQ_ABCD = """Output Contract (highest priority):
- Ignore any previous output-style instructions.
- Return exactly one option in this format: [A] or [B] or [C] or [D].
- Do not output any other text."""

EMBEDDED_MAIN2_OPEN = """Output Contract (highest priority):
- Ignore any previous output-style instructions.
- Return exactly one short final answer text only.
- Do not output any other text."""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

class SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _escape(s: str) -> str:
    """转义 { }，防止 format_map 误解析 story/JSON 里的花括号。"""
    return s.replace("{", "{{").replace("}", "}}")


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


def _expand_bracket_fields(template: str, row: Dict[str, Any]) -> str:
    """替换 {Story[full_story]} 形式的嵌套字段。"""
    def repl(m: re.Match) -> str:
        obj = row.get(m.group(1))
        return _escape(to_json_text(obj.get(m.group(2)) if isinstance(obj, dict) else None))
    return re.sub(r"\{([A-Za-z_]\w*)\[([^\]]+)\]\}", repl, template)


def _expand_top_level(template: str, row: Dict[str, Any]) -> str:
    """替换 {Story}、{Question} 等顶层字段。"""
    row_keys = set(row.keys())
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in row_keys:
            return m.group(0)
        if key == "Story":
            return _escape(_extract_story_text(row.get("Story")))
        return _escape(to_json_text(row.get(key)))
    return re.sub(r"\{([A-Za-z_]\w*)\}", repl, template)


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def build_prompt(
    template: str,
    row: Dict[str, Any],
    meta: SampleMeta,
    options_block: str = "",
) -> str:
    text = _expand_bracket_fields(template, row)
    text = _expand_top_level(text, row)
    row_meta = row.get("Meta", {})
    fields = SafeFormatDict(
        story=_escape(_extract_story_text(row.get("Story"))),
        question=_escape(str(row.get("Question", "")).strip()),
        action=_escape(to_json_text(row.get("Action", {}))),
        state=_escape(to_json_text(row.get("State", {}))),
        meta=_escape(to_json_text(row_meta)),
        answer=_escape(to_json_text(row.get("Answer"))),
        dataset=_escape(meta.dataset),
        split=_escape(meta.split),
        sample_id=_escape(str(row_meta.get("id", "")) if isinstance(row_meta, dict) else ""),
        options_block=_escape(options_block),
    )
    return text.format_map(fields)


def combine_prompt_three_stage(head: str, template: str, tail: str) -> str:
    return f"{head.strip()}\n\n---\n\n{template.strip()}\n\n---\n\n{tail.strip()}"


def load_main_prompts(prompt_dir: Path) -> Tuple[str, str, str, str]:
    def _read(fname: str, fallback: str) -> str:
        p = prompt_dir / fname
        return p.read_text(encoding="utf-8").strip() if p.exists() else fallback
    return (
        _read("main_mcq_abcd.txt", EMBEDDED_MAIN_MCQ_ABCD),
        _read("main_open.txt", EMBEDDED_MAIN_OPEN),
        _read("main2_mcq_abcd.txt", EMBEDDED_MAIN2_MCQ_ABCD),
        _read("main2_open.txt", EMBEDDED_MAIN2_OPEN),
    )


def load_templates(prompt_dir: Path, selected: Optional[List[str]]) -> Dict[str, str]:
    templates: Dict[str, str] = {}
    if prompt_dir.exists():
        for txt in sorted(prompt_dir.glob("*.txt")):
            if txt.stem.startswith("main_"):
                continue
            if selected and txt.stem not in selected:
                continue
            content = txt.read_text(encoding="utf-8").strip()
            if content:
                templates[txt.stem] = content
            else:
                print(f"[WARN] Skip empty template: {txt}")
    if not templates:
        print("[WARN] No usable template found. Using built-in default.")
        templates["default"] = DEFAULT_TEMPLATE
    return templates


def build_mcq_option_pack(
    row: Dict[str, Any], rng: random.Random
) -> Optional[Tuple[str, str, str]]:
    """
    尝试将样本构造为 4 选项 MCQ。
    返回 (options_block, gold_letter, question_stem)，无法构造则返回 None。
    """
    letters = ["A", "B", "C", "D"]
    correct_list = extract_answers(row.get("Answer"))
    wrong_list = extract_wrong_answers(row.get("Answer"))

    # 路径 1：结构化 correct/wrong（Tomato 等）
    if len(correct_list) == 1 and len(wrong_list) == 3:
        opts = [correct_list[0]] + wrong_list[:3]
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold_letter = letters[order.index(0)]
        lines = [f"{letters[i]}) {shuffled[i]}" for i in range(4)]
        return "Options:\n" + "\n".join(lines), gold_letter, str(row.get("Question", "")).strip()

    # 路径 2：选项嵌入 question 文本（ToMBench 等）
    q = str(row.get("Question", ""))
    pat = re.compile(r"\b([A-D])\s*[\.．、\):：]\s*", flags=re.IGNORECASE)
    ms = list(pat.finditer(q))
    for i in range(len(ms) - 3):
        if [ms[i+j].group(1).upper() for j in range(4)] != ["A", "B", "C", "D"]:
            continue
        chunk = ms[i:i+4]
        opts = [q[chunk[j].end(): (chunk[j+1].start() if j < 3 else len(q))].strip() for j in range(4)]
        if not all(opts):
            continue
        gold_raw = str(correct_list[0]).strip().upper() if correct_list else ""
        if gold_raw not in letters:
            continue
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold_letter = letters[order.index(letters.index(gold_raw))]
        lines = [f"{letters[i]}) {shuffled[i]}" for i in range(4)]
        return "Options:\n" + "\n".join(lines), gold_letter, q[:chunk[0].start()].strip()

    return None


def extract_answer_bracket_abcd(response: str) -> str:
    """从模型输出中提取 [A/B/C/D]。"""
    for pat in (
        r"final\s*answer\s*[:：]\s*\[\s*([ABCD])\s*\]",
        r"\[\s*([ABCD])\s*\]",
        r"final\s*answer\s*[:：]\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ):
        m = re.search(pat, response, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""
