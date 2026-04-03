#!/usr/bin/env python3
"""ToM baseline evaluation — 主循环，对接 vllm serve（OpenAI 兼容接口）。"""
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm import LLMClient

from data import SampleMeta, extract_answers, load_dataset_splits, normalize_tom_row
from prompt import (
    build_mcq_option_pack, build_prompt, combine_prompt_three_stage,
    extract_answer_bracket_abcd, load_main_prompts, load_templates,
)
from scoring import normalize_text, score_prediction


# ---------------------------------------------------------------------------
# Prompt 准备 + 打分（与 LLM 调用解耦）
# ---------------------------------------------------------------------------

@dataclass
class SampleJob:
    """一个样本的元信息 + 构建好的 prompt，等待批量推理。"""
    prompt: str
    template_name: str
    meta: SampleMeta
    shuffle_k: int
    answers: List[str]
    gold_letter: str        # MCQ 模式下的正确选项字母，否则为空
    scoring_mode_hint: str  # "bracket_abcd" | "" (空表示交给 score_prediction 推断)
    row: Dict[str, Any]     # 用于打分时的 infer_scoring_mode
    sample_id: str
    question: str


def prepare_prompt(
    template_name: str,
    template: str,
    row: Dict[str, Any],
    meta: SampleMeta,
    prompt_style: str,
    rng: random.Random,
    shuffle_k: int,
    main_mcq: str, main_open: str,
    main2_mcq: str, main2_open: str,
) -> SampleJob:
    """纯 CPU 操作：规范化数据、构建 prompt，返回等待推理的 SampleJob。"""
    row = normalize_tom_row(row)
    answers = extract_answers(row.get("Answer"))
    row_meta = row.get("Meta", {})
    sample_id = str(row_meta.get("id", "") or row_meta.get("Index", "")) if isinstance(row_meta, dict) else ""

    if prompt_style == "legacy":
        prompt = build_prompt(template, row, meta)
        return SampleJob(
            prompt=prompt, template_name=template_name, meta=meta,
            shuffle_k=shuffle_k, answers=answers, gold_letter="",
            scoring_mode_hint="", row=row, sample_id=sample_id,
            question=str(row.get("Question", "")),
        )

    pack = build_mcq_option_pack(row, rng)
    if pack is not None:
        options_block, gold_letter, q_stem = pack
        full = combine_prompt_three_stage(main_mcq, template, main2_mcq)
        prompt_row = {**row, "Question": q_stem} if q_stem else row
        prompt = build_prompt(full, prompt_row, meta, options_block=options_block)
        return SampleJob(
            prompt=prompt, template_name=template_name, meta=meta,
            shuffle_k=shuffle_k, answers=answers, gold_letter=gold_letter,
            scoring_mode_hint="bracket_abcd", row=row, sample_id=sample_id,
            question=str(row.get("Question", "")),
        )

    full = combine_prompt_three_stage(main_open, template, main2_open)
    prompt = build_prompt(full, row, meta)
    return SampleJob(
        prompt=prompt, template_name=template_name, meta=meta,
        shuffle_k=shuffle_k, answers=answers, gold_letter="",
        scoring_mode_hint="", row=row, sample_id=sample_id,
        question=str(row.get("Question", "")),
    )


def score_response(job: SampleJob, pred: str) -> Dict[str, Any]:
    """纯 CPU 操作：根据预测文本和 job 元信息打分，返回结果记录。"""
    pred_letter = ""
    if job.scoring_mode_hint == "bracket_abcd":
        pred_letter = extract_answer_bracket_abcd(pred)
        hit = bool(pred_letter and pred_letter == job.gold_letter)
        scoring_mode = "bracket_abcd"
    else:
        pred_n = normalize_text(pred)
        hit, scoring_mode = score_prediction(
            pred_n, [normalize_text(a) for a in job.answers],
            job.meta.dataset, job.row,
        )
    return {
        "template": job.template_name, "dataset": job.meta.dataset, "split": job.meta.split,
        "index": job.meta.index, "sample_id": job.sample_id,
        "question": job.question, "gold_answers": job.answers,
        "prediction": pred, "correct": hit, "scoring_mode": scoring_mode,
        "prompt_style": "legacy" if not job.scoring_mode_hint and not job.gold_letter else "two_layer",
        "shuffle_repeat": job.shuffle_k,
        "gold_letter": job.gold_letter, "pred_letter": pred_letter,
    }


# ---------------------------------------------------------------------------
# 输出工具
# ---------------------------------------------------------------------------

class ExperimentLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("w", encoding="utf-8")

    def write(self, msg: str, echo: bool = True) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fp.write(f"[{ts}] {msg}\n")
        self._fp.flush()
        if echo:
            print(msg, flush=True)

    def write_file_only(self, msg: str) -> None:
        self.write(msg, echo=False)

    def close(self) -> None:
        self._fp.close()


def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def build_summary_table(
    summary_records: List[Dict[str, Any]], column_mode: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if column_mode == "dataset":
        buckets: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {"total": 0, "correct": 0})
        for r in summary_records:
            buckets[(r["template"], r["dataset"])]["total"] += r["total"]
            buckets[(r["template"], r["dataset"])]["correct"] += r["correct"]
        col_keys = sorted({ds for _, ds in buckets})
        settings = sorted({tmpl for tmpl, _ in buckets})
        lookup = {(t, d): v["correct"] / v["total"] if v["total"] else math.nan
                  for (t, d), v in buckets.items()}
        return col_keys, [{"setting": s, **{c: lookup.get((s, c)) for c in col_keys}} for s in settings]

    col_tuples = sorted({(r["dataset"], r["split"]) for r in summary_records})
    col_keys = [f"{d}/{s}" for d, s in col_tuples]
    settings = sorted({r["template"] for r in summary_records})
    lookup_ds = {(r["template"], r["dataset"], r["split"]): float(r["accuracy"]) for r in summary_records}
    return col_keys, [
        {"setting": s, **{f"{d}/{sp}": lookup_ds.get((s, d, sp)) for d, sp in col_tuples}}
        for s in settings
    ]


def format_markdown_table(col_keys: List[str], table_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "| setting | " + " | ".join(col_keys) + " |",
        "|---|" + "|".join(["---"] * len(col_keys)) + "|",
    ]
    for row in table_rows:
        cells = [
            "" if (v := row.get(k)) is None or (isinstance(v, float) and math.isnan(v))
            else f"{float(v):.4f}"
            for k in col_keys
        ]
        lines.append("| " + row["setting"] + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ToM baseline evaluation via vLLM serve.")
    p.add_argument("--dataset-root", default="TomDatasets")
    p.add_argument("--prompt-dir",   default="prompt")
    p.add_argument("--result-dir",   default="result")
    p.add_argument("--predictions-jsonl", default=None)

    # API
    p.add_argument("--model",     required=True, help="vllm serve 注册的模型名。")
    p.add_argument("--model-tag", default=None,  help="显示名，默认同 --model。")
    p.add_argument("--api-url",   default="http://localhost:8000/v1")
    p.add_argument("--api-key",   default="not-needed")

    # 生成参数
    p.add_argument("--max-new-tokens", type=int,   default=2048)
    p.add_argument("--temperature",    type=float, default=0.01)
    p.add_argument("--top-p",          type=float, default=0.95)

    # 过滤
    p.add_argument("--max-samples-per-split", type=int, default=0, help="0 = 全量")
    p.add_argument("--dataset-filter",  nargs="*", default=None)
    p.add_argument("--split-filter",    nargs="*", default=["test"])
    p.add_argument("--include-all-splits", action="store_true")
    p.add_argument("--prompt-names",    nargs="*", default=None)

    # 评测设置
    p.add_argument("--summary-columns", choices=("dataset_split", "dataset"), default="dataset_split")
    p.add_argument("--prompt-style",    choices=("two_layer", "legacy"),       default="two_layer")
    p.add_argument("--shuffle-repeats", type=int, default=5)
    p.add_argument("--shuffle-base-seed", type=int, default=42)
    p.add_argument("--eval-phase",      choices=("none", "screen", "final"),   default="none")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.eval_phase == "screen" and args.max_samples_per_split == 0:
        args.max_samples_per_split = 50
        print("[INFO] eval_phase=screen: max-samples-per-split 默认为 50。", flush=True)

    dataset_root = Path(args.dataset_root)
    result_dir   = Path(args.result_dir)
    model_tag    = args.model_tag or args.model

    assert dataset_root.exists(), f"Dataset root not found: {dataset_root}"

    main_mcq, main_open, main2_mcq, main2_open = load_main_prompts(Path(args.prompt_dir))
    templates = load_templates(Path(args.prompt_dir), args.prompt_names)

    runner = LLMClient(
        model_name=args.model, api_key=args.api_key, api_url=args.api_url,
        temperature=args.temperature, max_tokens=args.max_new_tokens,
        top_p=args.top_p, enable_thinking=False,
    )
    print(f"[INFO] Model: {model_tag} @ {args.api_url}", flush=True)
    print(f"[INFO] {len(templates)} template(s). prompt_style={args.prompt_style} "
          f"shuffle_repeats={args.shuffle_repeats}", flush=True)

    splits = load_dataset_splits(dataset_root)
    if not splits:
        raise RuntimeError(f"No datasets found under {dataset_root}")

    all_records: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, Any]] = {}
    exp_log: Optional[ExperimentLog] = None

    try:
        exp_log = ExperimentLog(result_dir / "experiment.log")

        for dataset_name, split_name, rows in splits:
            if args.dataset_filter and dataset_name not in set(args.dataset_filter):
                continue
            if not args.include_all_splits:
                allowed = set(args.split_filter) if args.split_filter else {"test"}
                if split_name not in allowed:
                    continue
            if args.max_samples_per_split > 0:
                rows = rows[: args.max_samples_per_split]

            n_shuffles = max(1, args.shuffle_repeats) if args.prompt_style == "two_layer" else 1
            ds_hash = int(hashlib.md5(dataset_name.encode()).hexdigest()[:8], 16)

            for tmpl_name, template in templates.items():
                key = f"{tmpl_name}:{dataset_name}:{split_name}"
                summary.setdefault(key, {
                    "template": tmpl_name, "setting": tmpl_name,
                    "dataset": dataset_name, "split": split_name,
                    "total": 0, "correct": 0,
                })

                # ---- 1. 批量构建 prompt（纯 CPU）----
                jobs: List[SampleJob] = []
                for shuffle_k in range(n_shuffles):
                    for idx, row in enumerate(rows):
                        rng = random.Random(
                            args.shuffle_base_seed + shuffle_k * 1_000_003 + idx * 97 + (ds_hash % 100_009)
                        )
                        jobs.append(prepare_prompt(
                            template_name=tmpl_name, template=template,
                            row=row, meta=SampleMeta(dataset_name, split_name, idx),
                            prompt_style=args.prompt_style, rng=rng, shuffle_k=shuffle_k,
                            main_mcq=main_mcq, main_open=main_open,
                            main2_mcq=main2_mcq, main2_open=main2_open,
                        ))

                # ---- 2. 一次 batch_generate（并行推理）----
                exp_log.write(f"[BATCH] {tmpl_name} | {dataset_name}/{split_name} | {len(jobs)} prompts")
                batch_results = runner.batch_generate([j.prompt for j in jobs])

                # ---- 3. 批量打分 ----
                for job, (gens, _) in zip(jobs, batch_results):
                    pred = gens[0].text if gens else ""
                    rec = score_response(job, pred)
                    all_records.append(rec)
                    summary[key]["total"] += 1
                    summary[key]["correct"] += int(rec["correct"])
                    pred_tail = pred.replace("\n", "\\n")[-50:]
                    exp_log.write_file_only(
                        f"[STEP] {tmpl_name} | {dataset_name}/{split_name} | "
                        f"shuffle={rec['shuffle_repeat']} | idx={rec['index']} | "
                        f"{'OK' if rec['correct'] else 'FAIL'} | ...{pred_tail}"
                    )

                tot, cor = summary[key]["total"], summary[key]["correct"]
                acc = cor / tot if tot else float("nan")
                summary[key]["accuracy"] = acc
                exp_log.write(f"[UNIT_DONE] {tmpl_name} | {dataset_name}/{split_name} | "
                              f"accuracy={acc:.4f} | correct={cor}/{tot}")

                # 追加到 baseline.txt
                baseline_txt = result_dir / "baseline.txt"
                baseline_txt.parent.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with baseline_txt.open("a", encoding="utf-8") as f:
                    f.write(f"[{ts}] model={model_tag} | template={tmpl_name} | "
                            f"{dataset_name}/{split_name} | accuracy={acc:.4f} | correct={cor}/{tot}\n")

        # 汇总输出
        summary_records = sorted(summary.values(), key=lambda x: (x["template"], x["dataset"], x["split"]))
        col_keys, table_rows = build_summary_table(summary_records, args.summary_columns)
        table_md = format_markdown_table(col_keys, table_rows)

        table_doc = (
            f"# Baseline results\n\n"
            f"- **Model:** {model_tag} (`{args.model}` @ {args.api_url})\n"
            f"- **Prompt style:** `{args.prompt_style}`\n"
            f"- **Shuffle repeats:** {args.shuffle_repeats}\n"
            f"- **Eval phase:** `{args.eval_phase}`\n"
            f"- **Generated:** {datetime.now().isoformat()}\n"
            f"- **Inference calls:** {len(all_records)}\n\n"
            f"## Accuracy\n\n{table_md}\n\n"
            f"## Raw summary\n\n```json\n{json.dumps(summary_records, ensure_ascii=False, indent=2)}\n```\n"
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "results_table.md").write_text(table_doc, encoding="utf-8")

        if args.predictions_jsonl:
            save_jsonl(Path(args.predictions_jsonl), all_records)
            print(f"[DONE] Predictions: {args.predictions_jsonl}", flush=True)

        print(f"[DONE] Log: {result_dir}/experiment.log | Table: {result_dir}/results_table.md", flush=True)
        print(table_md, flush=True)

    finally:
        if exp_log:
            exp_log.close()


if __name__ == "__main__":
    main()
