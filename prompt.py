"""Prompt 构建：模板加载、字段填充、MCQ 选项打乱。"""
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
2) This question has {option_count} answer options labeled {option_letters_plain}.
3) Put only the option letter (one of {option_letters_plain}) inside XML-style tags, with no other text inside the tags:
   <answer>LETTER</answer>
4) Outside of <answer>...</answer> you may follow the middle template for reasoning style, but the model's final commitment must appear in that single tag pair."""

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
- You may provide reasoning according to the template prompt before the final tag.
- Your final line must be exactly one tag pair: <answer>LETTER</answer>
  where LETTER is exactly one of {option_letters_plain} (option letter only, no punctuation, no option text inside the tag)."""

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


CHOICE_LETTERS = tuple("ABCD")


@dataclass
class MCQPack:
    options_block: str
    gold_letter: str
    question_stem: str
    option_letters: Tuple[str, ...]


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


def _normalize_choice_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return s


def resolve_gold_letter(
    correct_text: str,
    option_letters: Sequence[str],
    option_texts: Sequence[str],
) -> str:
    """
    优先解析显式字母；若无字母，则按答案文本与选项文本对齐映射到字母。
    """
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


def build_mcq_prompt_fields(option_letters: Sequence[str]) -> Dict[str, str]:
    bracket_choices = [f"[{x}]" for x in option_letters]
    return {
        "option_count": str(len(option_letters)),
        "option_letters_plain": ", ".join(option_letters),
        "option_bracket_choices": ", ".join(bracket_choices),
        "option_bracket_or": " or ".join(bracket_choices),
    }


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
    extra_fields: Optional[Dict[str, str]] = None,
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
    if extra_fields:
        for k, v in extra_fields.items():
            fields[k] = _escape(str(v))
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
            if txt.stem.startswith("main_") or txt.stem.startswith("v2_"):
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
) -> Optional[MCQPack]:
    """
    将样本尽量构造为 2/3/4 选项 MCQ。
    返回 MCQPack；无法构造则返回 None。
    """
    correct_list = extract_answers(row.get("Answer"))
    wrong_list = extract_wrong_answers(row.get("Answer"))

    # 路径 1：结构化 correct/wrong（支持 2/3/4 选项）
    if len(correct_list) == 1 and 1 <= len(wrong_list) <= 3:
        option_letters = CHOICE_LETTERS[: 1 + len(wrong_list)]
        opts = [correct_list[0]] + wrong_list[: len(option_letters) - 1]
        order = list(range(len(option_letters)))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold_letter = option_letters[order.index(0)]
        lines = [f"{option_letters[i]}) {shuffled[i]}" for i in range(len(option_letters))]
        return MCQPack(
            options_block="Options:\n" + "\n".join(lines),
            gold_letter=gold_letter,
            question_stem=str(row.get("Question", "")).strip(),
            option_letters=tuple(option_letters),
        )

    # 路径 2：选项嵌入 question 文本（支持 2/3/4 选项；兼容 ToMBench）
    q = str(row.get("Question", ""))
    pat = re.compile(r"\b([A-D])\s*[\.．、\):：]\s*", flags=re.IGNORECASE)
    ms = list(pat.finditer(q))

    def _first_after(matches: List[re.Match], letter: str, start_pos: int) -> Optional[re.Match]:
        for m in matches:
            if m.start() > start_pos and m.group(1).upper() == letter:
                return m
        return None

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

        order = list(range(len(option_letters)))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold_letter = option_letters[order.index(option_letters.index(gold_raw))]
        lines = [f"{option_letters[i]}) {shuffled[i]}" for i in range(len(option_letters))]
        return MCQPack(
            options_block="Options:\n" + "\n".join(lines),
            gold_letter=gold_letter,
            question_stem=q[: chosen[0].start()].strip(),
            option_letters=tuple(option_letters),
        )

    return None


# ---------------------------------------------------------------------------
# ToMBench / Tomato v2：模板自 prompt/v2_*.txt 加载
# ---------------------------------------------------------------------------

_V2_PROMPT_DIR = Path(__file__).resolve().parent / "prompt"


def _read_v2_txt(filename: str) -> str:
    p = _V2_PROMPT_DIR / filename
    if not p.is_file():
        raise FileNotFoundError(f"v2 prompt file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def tombench_system_prompt(language: str, cot: bool) -> str:
    if language == "zh":
        name = "v2_tombench_system_zh_cot.txt" if cot else "v2_tombench_system_zh.txt"
    else:
        name = "v2_tombench_system_en_cot.txt" if cot else "v2_tombench_system_en.txt"
    return _read_v2_txt(name)


def tombench_user_prompt(
    user_language: str,
    num_options: int,
    story: str,
    question: str,
    choices_shuffled: List[str],
    task_type: str,
) -> str:
    """user_language: 用户模板语言 zh/en（可与系统提示语言独立组合）。"""
    lang = "zh" if user_language == "zh" else "en"
    if num_options >= 4:
        tpl = _read_v2_txt(f"v2_tombench_user_4_{lang}.txt")
        return tpl.format(
            story=story,
            question=question,
            choice_a=choices_shuffled[0],
            choice_b=choices_shuffled[1],
            choice_c=choices_shuffled[2],
            choice_d=choices_shuffled[3],
            task_type=task_type,
        )
    if num_options == 3:
        tpl = _read_v2_txt(f"v2_tombench_user_3_{lang}.txt")
        return tpl.format(
            story=story,
            question=question,
            choice_a=choices_shuffled[0],
            choice_b=choices_shuffled[1],
            choice_c=choices_shuffled[2],
            task_type=task_type,
        )
    tpl = _read_v2_txt(f"v2_tombench_user_2_{lang}.txt")
    return tpl.format(
        story=story,
        question=question,
        choice_a=choices_shuffled[0],
        choice_b=choices_shuffled[1],
        task_type=task_type,
    )


def tomato_mcqa_system_prompt() -> str:
    return _read_v2_txt("v2_tomato_system.txt")


def tomato_user_block(story_block: str, question: str, options: Dict[str, str]) -> str:
    """options: {'A': text, 'B': text, ...} 按字母顺序展示。"""
    lines_o = [f"[{letter}] {options[letter]}" for letter in sorted(options.keys())]
    options_block = "\n".join(lines_o)
    tpl = _read_v2_txt("v2_tomato_user.txt")
    fields = SafeFormatDict(
        story_block=_escape(story_block.strip()),
        question=_escape(question.strip()),
        options_block=_escape(options_block),
    )
    return tpl.format_map(fields)
