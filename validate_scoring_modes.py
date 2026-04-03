#!/usr/bin/env python3
"""
对 DEFAULT_DATASET_SCORING 中的每个数据集抽样跑 LLM，打印 gold / 预测 / 判题结果，
用于检查判题模式是否过严、过松或 gold 格式与模式不一致。

用法:
  python validate_scoring_modes.py --dataset-root TomDatasets --prompt-name "Agentic-ToM"
  python validate_scoring_modes.py --dry-run   # 不加载模型，只打印各数据集首条样本的 Answer 结构
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from run import (
    DEFAULT_DATASET_SCORING,
    QwenRunner,
    SampleMeta,
    build_prompt,
    extract_answers,
    extract_mcq_letter,
    infer_scoring_mode,
    iter_arrow_rows,
    load_templates,
    normalize_text,
    normalize_tom_row,
    open_substring_matches,
    score_prediction,
    _is_mcq_letter_gold,
)

PROJECT = Path(__file__).resolve().parent


def find_test_arrow(dataset_root: Path, dataset: str) -> Optional[Path]:
    sub = dataset_root / dataset / "test"
    if not sub.is_dir():
        return None
    arrows = sorted(sub.glob("data-*.arrow"))
    return arrows[0] if arrows else None


def sample_indices(n_rows: int, k: int, seed: int) -> List[int]:
    if n_rows <= k:
        return list(range(n_rows))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), k))


def print_row_diagnostics(
    dataset: str,
    row: Dict[str, Any],
    pred: str,
    scoring_mode_resolved: str,
    ok: bool,
) -> None:
    raw_ans = row.get("Answer")
    gold_list = extract_answers(raw_ans)
    pred_n = normalize_text(pred)
    answers_n = [normalize_text(a) for a in gold_list]
    ext = extract_mcq_letter(pred_n)

    print(f"  --- raw Answer (repr) ---")
    print(f"    {raw_ans!r}")
    print(f"  --- extract_answers -> normalize ---")
    for i, a in enumerate(answers_n):
        flag = _is_mcq_letter_gold(a) if a else False
        print(f"    gold[{i}]={a!r}  single_letter_mcq_gold={flag}")
    print(f"  infer_scoring_mode (preset) = {infer_scoring_mode(dataset, row)!r}")
    print(f"  scoring_mode (resolved)     = {scoring_mode_resolved!r}")
    print(f"  extract_mcq_letter(pred_n)= {ext!r}")
    if scoring_mode_resolved == "mcq_letter":
        print("  （开放式子串对单字母 gold 极易因英文词含 d/a 等误报，仅作参考）")
    else:
        print(
            f"  open_substring vs gold[0]   = "
            f"{open_substring_matches(pred_n, answers_n[0]) if answers_n else 'n/a'}"
        )
    print(f"  correct = {ok}")


def run_validation(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"[ERR] dataset-root 不存在: {dataset_root}", file=sys.stderr)
        return 1

    prompt_dir = Path(args.prompt_dir)
    templates = load_templates(prompt_dir, [args.prompt_name] if args.prompt_name else None, emit=print)
    if args.prompt_name:
        if args.prompt_name not in templates:
            print(f"[ERR] 未找到模板 {args.prompt_name!r}", file=sys.stderr)
            return 1
        template_name = args.prompt_name
        template_body = templates[template_name]
    else:
        template_name = sorted(templates.keys())[0]
        template_body = templates[template_name]

    print("=" * 72)
    print(f"模板: {template_name}")
    print(f"抽样: 每数据集 {args.samples} 条 | seed={args.seed}")
    print("=" * 72)

    runner: Optional[QwenRunner] = None
    if not args.dry_run:
        print("加载模型…", flush=True)
        runner = QwenRunner(
            model_name_or_path=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print("模型就绪。", flush=True)

    datasets = list(DEFAULT_DATASET_SCORING.keys())
    if args.datasets:
        want = set(args.datasets)
        datasets = [d for d in datasets if d in want]
        for d in want:
            if d not in DEFAULT_DATASET_SCORING:
                print(f"[WARN] 未在 DEFAULT_DATASET_SCORING 中: {d}", file=sys.stderr)

    overall_ok = 0
    overall_n = 0

    for dataset in datasets:
        preset = DEFAULT_DATASET_SCORING[dataset]
        arrow = find_test_arrow(dataset_root, dataset)
        print()
        print("#" * 72)
        print(f"数据集: {dataset}  |  预设判题: {preset}")
        print("#" * 72)

        if arrow is None:
            print(f"  [SKIP] 未找到 {dataset_root / dataset / 'test' / 'data-*.arrow'}")
            continue

        rows = list(iter_arrow_rows(arrow))
        if not rows:
            print("  [SKIP] arrow 为空")
            continue

        if args.dry_run:
            row0 = normalize_tom_row(dict(rows[0]))
            raw_ans = row0.get("Answer")
            gl = extract_answers(raw_ans)
            an = [normalize_text(x) for x in gl]
            print(f"  arrow: {arrow}")
            print(f"  行数: {len(rows)}")
            print(f"  首条 Answer 原始: {raw_ans!r}")
            print(f"  首条 extract_answers: {gl}")
            print(f"  归一化 gold: {an}")
            print(f"  gold 是否全为单字母 A–J: {all(_is_mcq_letter_gold(x) for x in an if x)}")
            print(f"  infer_scoring_mode: {infer_scoring_mode(dataset, row0)!r}")
            continue

        assert runner is not None
        if args.indices is not None:
            idxs = [int(x) for x in args.indices.split(",") if x.strip()]
            for i in idxs:
                if i < 0 or i >= len(rows):
                    print(f"  [ERR] index {i} 超出 [0,{len(rows)-1}]")
                    return 1
        else:
            idxs = sample_indices(len(rows), args.samples, args.seed + hash(dataset) % 10000)
        for idx in idxs:
            raw_row = rows[idx]
            row = normalize_tom_row(dict(raw_row))
            meta = SampleMeta(
                dataset=dataset,
                split="test",
                source_file=str(arrow),
                index=idx,
            )
            prompt = build_prompt(template_body, row, meta)
            t0 = time.perf_counter()
            pred = runner.generate(prompt)
            elapsed = time.perf_counter() - t0

            pred_n = normalize_text(pred)
            answers_n = [normalize_text(a) for a in extract_answers(row.get("Answer"))]
            ok, mode = score_prediction(pred_n, answers_n, dataset, row)

            overall_n += 1
            overall_ok += int(ok)

            print()
            print(f"  [index {idx}] 耗时 {elapsed:.1f}s | OK={ok} | mode={mode}")
            q = str(row.get("Question", ""))[:200]
            print(f"  Question(节选): {q}{'…' if len(str(row.get('Question','')))>200 else ''}")
            print(f"  prediction(节选): {pred[:400]}{'…' if len(pred)>400 else ''}")
            print_row_diagnostics(dataset, row, pred, mode, ok)

    if not args.dry_run and overall_n:
        print()
        print("=" * 72)
        print(f"抽样合计: {overall_ok}/{overall_n} = {overall_ok/overall_n:.2%}")
        print("=" * 72)

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按数据集验证判题逻辑与 LLM 输出是否匹配")
    p.add_argument("--dataset-root", type=str, default=str(PROJECT / "TomDatasets"))
    p.add_argument("--prompt-dir", type=str, default=str(PROJECT / "prompt"))
    p.add_argument(
        "--prompt-name",
        type=str,
        default=None,
        help="prompt 文件名（不含 .txt）；默认取字典序第一个",
    )
    p.add_argument("--datasets", nargs="*", default=None, help="只跑指定数据集名，默认四个都跑")
    p.add_argument("--samples", type=int, default=3, help="每数据集抽样条数（与 --indices 互斥）")
    p.add_argument(
        "--indices",
        type=str,
        default=None,
        help='固定行号，逗号分隔，如 "0,1,2"（用于对照 experiment.log 前几条）',
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="不加载模型；只打印每个数据集 test 首条样本的 Answer 结构（需数据存在）",
    )
    p.add_argument("--model", type=str, default="/data/yugx/LongBench/simple_tune/Qwen3-4B")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_validation(parse_args()))
