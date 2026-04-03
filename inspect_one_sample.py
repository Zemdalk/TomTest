#!/usr/bin/env python3
"""
从某个 test 集里抽一条样本：用指定 prompt 走一遍 LLM，打印完整 prompt、模型输出、
标准答案及脚本内置的字符串判题结果，方便你人工判断「是否真对」。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run import (
    QwenRunner,
    SampleMeta,
    build_prompt,
    extract_answers,
    extract_mcq_letter,
    extract_story_text,
    infer_scoring_mode,
    iter_arrow_rows,
    load_templates,
    normalize_text,
    normalize_tom_row,
    open_substring_matches,
    score_against_gold,
    score_prediction,
    yes_no_matches,
)

PROJECT = Path(__file__).resolve().parent


def find_arrow(dataset_root: Path, dataset: str, split: str) -> Path:
    sub = dataset_root / dataset / split
    if not sub.is_dir():
        raise FileNotFoundError(f"目录不存在: {sub}")
    arrows = sorted(sub.glob("data-*.arrow"))
    if not arrows:
        raise FileNotFoundError(f"未找到 .arrow: {sub}")
    return arrows[0]


def pick_template(
    prompt_dir: Path, prompt_name: Optional[str], emit=print
) -> Tuple[str, str]:
    sel = [prompt_name] if prompt_name else None
    templates = load_templates(prompt_dir, sel, emit=emit)
    if prompt_name:
        if prompt_name not in templates:
            all_n = sorted(load_templates(prompt_dir, None, emit=lambda _: None).keys())
            raise SystemExit(
                f"未找到模板 {prompt_name!r}。已有: {', '.join(all_n)}"
            )
        return prompt_name, templates[prompt_name]
    name = sorted(templates.keys())[0]
    return name, templates[name]


def explain_auto_score(
    pred_n: str,
    answers_n: List[str],
    dataset: str,
    row: Dict[str, Any],
    scoring_mode: str,
) -> str:
    if not answers_n:
        return "数据里没有可解析的标准答案 → 脚本无法自动判对。"
    bits = [f"判题模式={scoring_mode}（与 run.py evaluate_one 一致；infer_scoring_mode 为 {infer_scoring_mode(dataset, row)!r}）"]
    for a in answers_n:
        if not a:
            continue
        ok = score_against_gold(pred_n, a, scoring_mode)
        if scoring_mode == "mcq_letter":
            got = extract_mcq_letter(pred_n)
            bits.append(f"选择题 gold「{a}」: 提取选项={got!r} → {ok}")
        elif scoring_mode == "open_substring":
            bits.append(
                f"开放式子串 gold「{a}」: {ok}（双向子串={open_substring_matches(pred_n, a)}）"
            )
        elif scoring_mode == "open_normalized":
            bits.append(f"开放式全等 gold「{a}」: {ok}")
        elif scoring_mode == "yes_no":
            bits.append(f"是否题 gold「{a}」: {ok}（yes_no_matches={yes_no_matches(pred_n, a)}）")
        else:
            bits.append(f"gold「{a}」: {ok}")
    return " | ".join(bits)


def main() -> None:
    p = argparse.ArgumentParser(description="单条 test 样本 + LLM + 与答案对照")
    p.add_argument("--dataset-root", type=str, default=str(PROJECT / "TomDatasets"))
    p.add_argument("--dataset", type=str, default="ToMi")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--index", type=int, default=0, help="该 split 内从 0 开始的行号")
    p.add_argument("--prompt-dir", type=str, default=str(PROJECT / "prompt"))
    p.add_argument(
        "--prompt-name",
        type=str,
        default="Standard Zero-shot",
        help="prompt 的 txt 文件名（不含 .txt），与 run.py 一致",
    )
    p.add_argument(
        "--model",
        type=str,
        default="/data/yugx/LongBench/simple_tune/Qwen3-4B",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="越大越慢；短答试跑可设 64",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    arrow = find_arrow(dataset_root, args.dataset, args.split)
    rows = list(iter_arrow_rows(arrow))
    if args.index < 0 or args.index >= len(rows):
        raise SystemExit(f"index={args.index} 超出范围 [0, {len(rows) - 1}]")
    raw_row: Dict[str, Any] = rows[args.index]
    row = normalize_tom_row(dict(raw_row))

    prompt_dir = Path(args.prompt_dir)
    template_name, template_body = pick_template(prompt_dir, args.prompt_name)

    meta = SampleMeta(
        dataset=args.dataset,
        split=args.split,
        source_file=str(arrow),
        index=args.index,
    )
    prompt = build_prompt(template_body, row, meta)

    print("=" * 72)
    print("【样本概要】")
    print(f"  文件: {arrow}")
    print(f"  index: {args.index} / {len(rows) - 1}")
    rmeta = row.get("Meta", {})
    if isinstance(rmeta, dict):
        print(f"  Meta: {json.dumps(rmeta, ensure_ascii=False)}")
    print(f"  Question: {row.get('Question', '')}")
    st = extract_story_text(row.get("Story"))
    print(f"  Story(节选): {st[:400]}{'…' if len(st) > 400 else ''}")
    print(f"  标准答案(原始): {extract_answers(row.get('Answer'))}")
    print("=" * 72)

    print("【选用模板】", template_name)
    print("-" * 72)
    print("【完整 Prompt】")
    print(prompt)
    print("-" * 72)

    print("加载模型…", flush=True)
    runner = QwenRunner(
        model_name_or_path=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("模型就绪。", flush=True)
    print(
        f"正在生成回复（max_new_tokens={args.max_new_tokens}，此阶段无进度条，"
        f"GPU 上常见需几秒～几十秒，CPU 会更慢）…",
        flush=True,
    )
    t0 = time.perf_counter()
    pred = runner.generate(prompt)
    elapsed = time.perf_counter() - t0
    print(f"生成结束，耗时 {elapsed:.1f}s。", flush=True)

    gold = extract_answers(row.get("Answer"))
    pred_n = normalize_text(pred)
    answers_n = [normalize_text(a) for a in gold]
    auto_ok, scoring_mode = score_prediction(pred_n, answers_n, args.dataset, row)

    print("【模型输出（原始）】")
    print(pred)
    print("=" * 72)
    print("【归一化后（与 run.py 判分一致）】")
    print(f"  scoring_mode: {scoring_mode}")
    print(f"  prediction: {pred_n!r}")
    for i, a in enumerate(answers_n):
        print(f"  gold[{i}]:   {a!r}")
    print("-" * 72)
    print("【脚本自动判题】", "OK" if auto_ok else "FAIL")
    print("【规则说明】", explain_auto_score(pred_n, answers_n, args.dataset, row, scoring_mode))
    print("=" * 72)
    print(
        "请根据上面「原始输出」与「标准答案」自行核对；"
        "自动 OK 表示在 run.py 为该数据集选用的判题逻辑下命中。"
    )


if __name__ == "__main__":
    main()
