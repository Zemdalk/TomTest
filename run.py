#!/usr/bin/env python3
"""ToM baseline evaluation — 主循环，对接 vllm serve（OpenAI 兼容接口）。"""
import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm import LLMClient

from data import SampleMeta, extract_answers, load_dataset_splits, normalize_tom_row, to_json_text
from prompt import (
    build_mcq_option_pack,
    build_mcq_prompt_fields,
    build_prompt,
    combine_prompt_three_stage,
    load_main_prompts,
    load_templates,
    tombench_system_prompt,
    tombench_user_prompt,
    tomato_mcqa_system_prompt,
    tomato_user_block,
)
from scoring import (
    create_position_map,
    extract_bracket_answer,
    extract_mcq_letter,
    extract_tomato_choice,
    extract_unshuffled_mcq,
    normalize_text,
    score_prediction,
)


MCQ_SHUFFLE_DATASETS = frozenset({"Tomato", "ToMBench"})
MCQ_VOTE_DATASETS = frozenset({"ToMBench"})
DEFAULT_JUDGE_OPEN_DATASETS = ("ToMi", "ToMQA")

OPEN_JUDGE_PROMPT = """You are a Theory of Mind (ToM) evaluation system. Your only function is to compare a model's answer against the ground truth and output whether it is correct.

## Task
Determine if the model demonstrates correct understanding of mental states (beliefs, intentions, desires, emotions, knowledge), including false beliefs and second-order attributions.

## Input
Context: {context}
Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}

## Strict Judgment Rules
Output ONLY ONE of these two strings, with no explanation, no reasoning, no preamble, and no markdown formatting:
- CORRECT
- INCORRECT

## Criteria for CORRECT
- The model answer matches the ground truth's mental state attribution (e.g., character believes X, feels Y, intends Z)
- For false belief questions: The model correctly identifies what the character falsely believes, not the reality
- For second-order questions: The model correctly identifies "A believes that B believes/feel X"
- Wording differences are allowed if the mental state attribution is identical

## Criteria for INCORRECT (including but not limited to)
- Reality bias: Model answers based on real facts instead of character's limited/false belief
- Order confusion: Mistakes first-order for second-order mental states
- Attribution error: Wrong mental state category (e.g., says "angry" when ground truth is "relieved")
- Ambiguity: Answer is vague, hedged, or contains multiple conflicting possibilities
- Egocentric bias: Assumes character knows information they don't have access to

## Output Format
Your response must contain EXACTLY ONE word:
CORRECT
or
INCORRECT
"""

OPEN_JUDGE_INSTRUCTION = "You are a strict Theory of Mind evaluation system. Reply with exactly one word: CORRECT or INCORRECT."


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
    option_letters: Tuple[str, ...] = ()
    option_count: int = 0


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
            question=str(row.get("Question", "")), option_letters=(), option_count=0,
        )

    pack = build_mcq_option_pack(row, rng)
    if pack is not None:
        full = combine_prompt_three_stage(main_mcq, template, main2_mcq)
        prompt_row = {**row, "Question": pack.question_stem} if pack.question_stem else row
        prompt = build_prompt(
            full,
            prompt_row,
            meta,
            options_block=pack.options_block,
            extra_fields=build_mcq_prompt_fields(pack.option_letters),
        )
        return SampleJob(
            prompt=prompt, template_name=template_name, meta=meta,
            shuffle_k=shuffle_k, answers=answers, gold_letter=pack.gold_letter,
            scoring_mode_hint="mcq_choice", row=row, sample_id=sample_id,
            question=str(row.get("Question", "")),
            option_letters=pack.option_letters, option_count=len(pack.option_letters),
        )

    full = combine_prompt_three_stage(main_open, template, main2_open)
    prompt = build_prompt(full, row, meta)
    return SampleJob(
        prompt=prompt, template_name=template_name, meta=meta,
        shuffle_k=shuffle_k, answers=answers, gold_letter="",
        scoring_mode_hint="", row=row, sample_id=sample_id,
        question=str(row.get("Question", "")), option_letters=(), option_count=0,
    )


def score_response(job: SampleJob, pred: str) -> Dict[str, Any]:
    """纯 CPU 操作：根据预测文本和 job 元信息打分，返回结果记录。"""
    pred_letter = ""
    if job.scoring_mode_hint == "mcq_choice":
        pred_letter = (extract_mcq_letter(pred, job.option_letters) or "").upper()
        hit = bool(pred_letter and pred_letter == job.gold_letter)
        scoring_mode = f"mcq_choice_{job.option_count}"
    else:
        hit, scoring_mode = score_prediction(
            pred, [normalize_text(a) for a in job.answers],
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


def _should_shuffle(dataset_name: str, prompt_style: str, shuffle_repeats: int) -> int:
    if prompt_style != "two_layer":
        return 1
    return max(1, shuffle_repeats) if dataset_name in MCQ_SHUFFLE_DATASETS else 1


def _pred_tail_suffix(pred: str, tail_len: int) -> str:
    """模型原文末尾若干字符（换行转为 \\n），tail_len<=0 表示不记录。"""
    if tail_len <= 0:
        return ""
    s = (pred or "").replace("\n", "\\n")
    return s[-tail_len:] if len(s) >= tail_len else s


def _log_step_line(exp_log: "ExperimentLog", msg: str, where: str) -> None:
    """where: experiment | stdout | both"""
    if where == "experiment":
        exp_log.write_file_only(msg)
    elif where == "stdout":
        print(msg, flush=True)
    elif where == "both":
        exp_log.write(msg, echo=True)
    else:
        raise ValueError(f"unknown log_pred_where: {where}")


def _build_open_judge_prompt(job: SampleJob, pred: str) -> str:
    context = to_json_text(
        {
            "Story": job.row.get("Story", {}),
            "Action": job.row.get("Action", {}),
            "State": job.row.get("State", {}),
            "Meta": job.row.get("Meta", {}),
        }
    )
    return OPEN_JUDGE_PROMPT.format(
        context=context,
        question=job.question,
        ground_truth=" | ".join(job.answers),
        model_answer=pred,
    )


def _parse_judge_verdict(text: str) -> Tuple[bool, str]:
    cleaned = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.IGNORECASE | re.DOTALL)
    upper = cleaned.strip().upper()
    if "INCORRECT" in upper:
        return False, "INCORRECT"
    if "CORRECT" in upper:
        return True, "CORRECT"
    return False, "INVALID"


def _score_standard_batch(
    jobs: List[SampleJob],
    batch_results: List[Tuple[List[Any], Any]],
    summary_item: Dict[str, Any],
    all_records: List[Dict[str, Any]],
    exp_log: "ExperimentLog",
    dataset_name: str,
    split_name: str,
    tmpl_name: str,
    log_pred_tail: int = 20,
    log_pred_where: str = "experiment",
) -> None:
    for job, (gens, _) in zip(jobs, batch_results):
        pred = gens[0].text if gens else ""
        rec = score_response(job, pred)
        all_records.append(rec)
        summary_item["total"] += 1
        summary_item["correct"] += int(rec["correct"])
        suf = _pred_tail_suffix(pred, log_pred_tail)
        tail_seg = f" | ...{suf}" if suf else ""
        _log_step_line(
            exp_log,
            f"[STEP] {tmpl_name} | {dataset_name}/{split_name} | "
            f"shuffle={rec['shuffle_repeat']} | idx={rec['index']} | "
            f"{'OK' if rec['correct'] else 'FAIL'}{tail_seg}",
            log_pred_where,
        )


def _score_voted_mcq_batch(
    jobs: List[SampleJob],
    batch_results: List[Tuple[List[Any], Any]],
    summary_item: Dict[str, Any],
    all_records: List[Dict[str, Any]],
    exp_log: "ExperimentLog",
    dataset_name: str,
    split_name: str,
    tmpl_name: str,
    log_pred_tail: int = 20,
    log_pred_where: str = "experiment",
) -> None:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for job, (gens, _) in zip(jobs, batch_results):
        pred = gens[0].text if gens else ""
        pred_letter = (extract_mcq_letter(pred, job.option_letters) or "").upper()
        suf = _pred_tail_suffix(pred, log_pred_tail)
        tail_seg = f" | ...{suf}" if suf else ""
        _log_step_line(
            exp_log,
            f"[STEP] {tmpl_name} | {dataset_name}/{split_name} | "
            f"shuffle={job.shuffle_k} | idx={job.meta.index} | "
            f"vote={pred_letter or '-'}{tail_seg}",
            log_pred_where,
        )
        grouped[(job.meta.index, job.sample_id)].append(
            {"job": job, "prediction": pred, "pred_letter": pred_letter}
        )

    for (_, _), items in grouped.items():
        first_job = items[0]["job"]
        votes: Dict[str, int] = {}
        for item in items:
            letter = item["pred_letter"]
            if not letter:
                continue
            if letter not in votes:
                votes[letter] = 0
            votes[letter] += 1
        winner = max(votes, key=votes.get) if votes else ""
        hit = bool(winner and winner == first_job.gold_letter)
        rec = {
            "template": first_job.template_name,
            "dataset": first_job.meta.dataset,
            "split": first_job.meta.split,
            "index": first_job.meta.index,
            "sample_id": first_job.sample_id,
            "question": first_job.question,
            "gold_answers": first_job.answers,
            "prediction": winner,
            "correct": hit,
            "scoring_mode": f"mcq_vote_{first_job.option_count}",
            "prompt_style": "two_layer",
            "shuffle_repeat": len(items),
            "gold_letter": first_job.gold_letter,
            "pred_letter": winner,
            "vote_counts": votes,
            "vote_trace": [x["pred_letter"] for x in items],
        }
        all_records.append(rec)
        summary_item["total"] += 1
        summary_item["correct"] += int(hit)
        _log_step_line(
            exp_log,
            f"[VOTE] {tmpl_name} | {dataset_name}/{split_name} | idx={first_job.meta.index} | "
            f"gold={first_job.gold_letter} | winner={winner or '-'} | votes={json.dumps(votes, ensure_ascii=False)} | "
            f"{'OK' if hit else 'FAIL'}",
            log_pred_where,
        )


def _score_judged_open_batch(
    jobs: List[SampleJob],
    batch_results: List[Tuple[List[Any], Any]],
    summary_item: Dict[str, Any],
    all_records: List[Dict[str, Any]],
    exp_log: "ExperimentLog",
    dataset_name: str,
    split_name: str,
    tmpl_name: str,
    judge_runner: LLMClient,
    log_pred_tail: int = 20,
    log_pred_where: str = "experiment",
) -> None:
    judge_prompts = []
    predictions = []
    for job, (gens, _) in zip(jobs, batch_results):
        pred = gens[0].text if gens else ""
        predictions.append(pred)
        judge_prompts.append(_build_open_judge_prompt(job, pred))

    exp_log.write(
        f"[BATCH_JUDGE] {tmpl_name} | {dataset_name}/{split_name} | {len(judge_prompts)} judge prompts"
    )
    judge_results = judge_runner.batch_generate(
        judge_prompts,
        instructions=[OPEN_JUDGE_INSTRUCTION] * len(judge_prompts),
    )

    for job, pred, (judge_gens, _) in zip(jobs, predictions, judge_results):
        judge_text = judge_gens[0].text if judge_gens else ""
        hit, judge_label = _parse_judge_verdict(judge_text)
        rec = {
            "template": job.template_name,
            "dataset": job.meta.dataset,
            "split": job.meta.split,
            "index": job.meta.index,
            "sample_id": job.sample_id,
            "question": job.question,
            "gold_answers": job.answers,
            "prediction": pred,
            "correct": hit,
            "scoring_mode": "llm_judge",
            "prompt_style": "two_layer",
            "shuffle_repeat": job.shuffle_k,
            "gold_letter": "",
            "pred_letter": "",
            "judge_model": judge_runner.model,
            "judge_verdict": judge_label,
            "judge_raw": judge_text,
        }
        all_records.append(rec)
        summary_item["total"] += 1
        summary_item["correct"] += int(hit)
        suf = _pred_tail_suffix(pred, log_pred_tail)
        tail_seg = f" | ...{suf}" if suf else ""
        _log_step_line(
            exp_log,
            f"[STEP] {tmpl_name} | {dataset_name}/{split_name} | idx={job.meta.index} | "
            f"judge={judge_label} | {'OK' if hit else 'FAIL'}{tail_seg}",
            log_pred_where,
        )


# ---------------------------------------------------------------------------
# 输出工具
# ---------------------------------------------------------------------------

class ExperimentLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("w", encoding="utf-8")

    @staticmethod
    def append_to_path(path: Path, msg: str) -> None:
        """与 write 相同时间戳格式，追加一行（供 v2 逐条日志等，无需持有 ExperimentLog 实例）。"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")

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
    p.add_argument("--judge-model", default=os.environ.get("TOMTEST_JUDGE_MODEL", "Qwen3-8B"))
    p.add_argument(
        "--judge-api-url",
        default=os.environ.get("TOMTEST_JUDGE_API_URL", "http://127.0.0.1:8010/v1"),
    )
    p.add_argument(
        "--judge-api-key",
        default=os.environ.get("TOMTEST_JUDGE_API_KEY", "not-needed"),
    )
    p.add_argument(
        "--judge-open-datasets",
        nargs="*",
        default=list(DEFAULT_JUDGE_OPEN_DATASETS),
        help="Datasets judged by external LLM-as-a-judge instead of string matching.",
    )

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
    p.add_argument(
        "--batch-chunk-size",
        type=int,
        default=256,
        metavar="N",
        help="按 N 条样本分块执行 生成+打分；>0 可在运行中持续产生日志，<=0 表示整批一次性处理。",
    )

    # 运行中查看模型原文（末尾片段）
    p.add_argument(
        "--log-pred-tail",
        type=int,
        default=20,
        metavar="N",
        help="[STEP] 附带模型输出末尾 N 个字符（换行记为 \\n）；0 表示不附带原文尾缀。",
    )
    p.add_argument(
        "--log-pred-where",
        choices=("experiment", "stdout", "both"),
        default="experiment",
        help="[STEP]/[VOTE] 行写入 experiment.log、仅 stdout（可进入 run.log）或两者。",
    )
    p.add_argument(
        "--backend",
        choices=("api", "hf"),
        default="api",
        help="api=OpenAI 兼容接口；hf=本地 HF 权重",
    )
    p.add_argument(
        "--hf-model-path",
        default=None,
        help="backend=hf 时必须提供",
    )
    p.add_argument("--device", default="cuda:0", help="backend=hf 时使用")
    p.add_argument(
        "--v2-language",
        choices=("zh", "en"),
        default="zh",
        help="ToMBench：系统提示语言",
    )
    p.add_argument(
        "--v2-user-language",
        choices=("zh", "en"),
        default=None,
        help="ToMBench：用户提示语言；默认与 --v2-language 相同",
    )
    p.add_argument("--v2-cot", action="store_true", help="ToMBench：启用 CoT")
    p.add_argument("--v2-shuffle-votes", type=int, default=5, help="ToMBench：投票次数")
    p.add_argument("--v2-seed", type=int, default=42, help="随机种子")
    p.add_argument(
        "--tomato-mode",
        choices=("mcqa", "genqa"),
        default="genqa",
        help="Tomato：mcqa=HF/NLL；genqa=生成式解析",
    )
    p.add_argument(
        "--v2-log-steps",
        action="store_true",
        help="ToMBench：逐条写 [STEP] 日志",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# v2：本地 HF 对话生成（原 eval_v2.hf_generate，内联以避免额外模块）
# ---------------------------------------------------------------------------


def load_hf_model(model_path: str, device: str) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tok


def hf_chat_generate(
    model: Any,
    tokenizer: Any,
    device: str,
    system: str,
    user: str,
    *,
    max_new_tokens: int = 1024,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError("Tokenizer has no chat_template; cannot run HF chat generation.")

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    new_tokens = out[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return (text or "").strip()


def run_tombench_v2(
    rows: List[Dict[str, Any]],
    generate_fn: Callable[[str, str], str],
    *,
    system_language: str,
    user_language: str,
    cot: bool,
    shuffle_votes: int,
    base_seed: int,
    backend: str,
    result_jsonl: Optional[Path] = None,
    experiment_log: Optional[Path] = None,
    log_steps: bool = False,
) -> Dict[str, Any]:
    """generate_fn(user_text, system_text) -> 模型原文。"""
    system = tombench_system_prompt(system_language, cot)
    records: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    fp = result_jsonl.open("w", encoding="utf-8") if result_jsonl else None
    pipeline = f"v2_tombench_{backend}"

    for idx, row in enumerate(rows):
        parsed = extract_unshuffled_mcq(row)
        if parsed is None:
            rec = {"index": idx, "skipped": True, "reason": "not_mcq"}
            records.append(rec)
            if fp:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if experiment_log and log_steps:
                ExperimentLog.append_to_path(experiment_log, f"[STEP] idx={idx} SKIP not_mcq")
            continue

        original_choices = parsed["original_choices"]
        gold = parsed["gold_letter"]
        num_opts = parsed["num_options"]
        story = parsed["story"]
        question = parsed["question"]
        task_type = parsed["task_type"]

        vote_letters: List[str] = []
        raw_last = ""
        letters = sorted(original_choices.keys())
        for k in range(shuffle_votes):
            rng = random.Random(base_seed + idx * 100_003 + k * 17)
            texts = [original_choices[x] for x in letters]
            shuffled_texts = texts.copy()
            rng.shuffle(shuffled_texts)
            pos_map = create_position_map(shuffled_texts, original_choices)
            user_txt = tombench_user_prompt(
                user_language, num_opts, story, question, shuffled_texts, task_type
            )
            raw = generate_fn(user_txt, system)
            raw_last = raw
            ext = extract_bracket_answer(raw)
            mapped: Optional[str] = None
            if ext and ext in pos_map:
                mapped = pos_map[ext]
            if mapped:
                vote_letters.append(mapped)

        if vote_letters:
            winner = Counter(vote_letters).most_common(1)[0][0]
        else:
            winner = ""

        hit = bool(winner and winner == gold)
        correct += int(hit)
        total += 1

        rec = {
            "index": idx,
            "skipped": False,
            "gold_letter": gold,
            "pred_letter": winner,
            "correct": hit,
            "vote_letters": vote_letters,
            "shuffle_votes": shuffle_votes,
            "system_language": system_language,
            "user_language": user_language,
            "cot": cot,
            "pipeline": pipeline,
            "backend": backend,
            "raw_last": (raw_last or "")[:2000],
        }
        records.append(rec)
        if fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if experiment_log and log_steps:
            tag = "OK" if hit else "FAIL"
            ExperimentLog.append_to_path(
                experiment_log,
                f"[STEP] idx={idx} gold={gold} pred={winner or '-'} {tag}",
            )

    if fp:
        fp.close()

    acc = correct / total if total else 0.0
    return {
        "dataset": "ToMBench",
        "pipeline": pipeline,
        "backend": backend,
        "system_language": system_language,
        "user_language": user_language,
        "cot": cot,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "records": records,
    }


def run_tomato_v2(
    rows: List[Dict[str, Any]],
    *,
    backend: str,
    eval_mode: str,
    result_jsonl: Optional[Path] = None,
    generate_fn: Optional[Callable[[str, str], str]] = None,
    model: Any = None,
    tokenizer: Any = None,
    device: str = "",
) -> Dict[str, Any]:
    """eval_mode: \"nll\"（HF 条件似然）或 \"generate\"（API/HF 生成式）。"""

    def _nll_assistant_suffix(messages: List[Dict[str, str]]) -> float:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(device)
        prompt_ids = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        plen = prompt_ids.shape[1]
        labels = input_ids.clone()
        labels[:, :plen] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
        return float(out.loss.item())

    records: List[Dict[str, Any]] = []
    correct = 0
    total = 0
    fp = result_jsonl.open("w", encoding="utf-8") if result_jsonl else None
    system = tomato_mcqa_system_prompt()
    if eval_mode == "nll":
        pipeline = f"v2_tomato_nll_{backend}"
    else:
        pipeline = f"v2_tomato_generate_{backend}"

    for idx, row in enumerate(rows):
        parsed = extract_unshuffled_mcq(row)
        if parsed is None:
            rec = {"index": idx, "skipped": True, "reason": "not_mcq"}
            records.append(rec)
            if fp:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        oc = parsed["original_choices"]
        gold = parsed["gold_letter"]
        story = parsed["story"]
        question = parsed["question"]
        user_body = tomato_user_block(story, question, oc)

        if eval_mode == "nll":
            scores: Dict[str, float] = {}
            for letter in sorted(oc.keys()):
                tag = f"[{letter}]"
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_body},
                    {"role": "assistant", "content": tag},
                ]
                scores[letter] = _nll_assistant_suffix(messages)
            pred = min(scores, key=scores.get)
            hit = pred == gold
            correct += int(hit)
            total += 1
            rec = {
                "index": idx,
                "skipped": False,
                "gold_letter": gold,
                "pred_letter": pred,
                "correct": hit,
                "scores": scores,
                "pipeline": pipeline,
                "backend": backend,
            }
        else:
            assert generate_fn is not None
            allowed = "".join(sorted(oc.keys()))
            raw = generate_fn(user_body, system)
            pred = extract_tomato_choice(raw, allowed) or ""
            hit = bool(pred and pred == gold)
            correct += int(hit)
            total += 1
            rec = {
                "index": idx,
                "skipped": False,
                "gold_letter": gold,
                "pred_letter": pred,
                "correct": hit,
                "raw": (raw or "")[:2000],
                "pipeline": pipeline,
                "backend": backend,
            }
        records.append(rec)
        if fp:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if fp:
        fp.close()

    acc = correct / total if total else 0.0
    return {
        "dataset": "Tomato",
        "pipeline": pipeline,
        "backend": backend,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "records": records,
    }


def _v2_generate_fn(
    args: argparse.Namespace,
    client: Optional[LLMClient],
    hf_bundle: Optional[Tuple[Any, Any]],
) -> Callable[[str, str], str]:
    """API 与 HF 共用的 (user, system) -> text 闭包。"""
    if args.backend == "api":
        assert client is not None

        def fn_api(user: str, system: str) -> str:
            gens, _ = client.generate(user, instruction=system, n=1)
            return (gens[0].text or "").strip() if gens else ""

        return fn_api
    assert hf_bundle is not None
    model, tokenizer = hf_bundle

    def fn_hf(user: str, system: str) -> str:
        return hf_chat_generate(
            model, tokenizer, args.device, system, user, max_new_tokens=args.max_new_tokens
        )

    return fn_hf


def _run_v2_pipeline(args: argparse.Namespace, result_dir: Path) -> None:
    root = Path(args.dataset_root)
    if not root.is_dir():
        raise RuntimeError(f"dataset-root not found: {root}")
    if args.backend == "hf" and not args.hf_model_path:
        raise RuntimeError("pipeline=v2 且 backend=hf 需要 --hf-model-path")

    v2_user_lang = args.v2_user_language if args.v2_user_language is not None else args.v2_language
    splits = load_dataset_splits(root)
    allowed_splits = set(args.split_filter or ["test"])
    wanted = set(args.dataset_filter) if args.dataset_filter else {"ToMBench", "Tomato"}

    result_dir.mkdir(parents=True, exist_ok=True)
    exp_log = result_dir / "experiment.log"
    exp_log.write_text("", encoding="utf-8")

    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with exp_log.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
        print(msg, flush=True)

    _log(
        f"[START] run.py pipeline=v2 backend={args.backend} model={args.model} api={args.api_url} "
        f"datasets={sorted(wanted)} sys={args.v2_language} user={v2_user_lang} cot={args.v2_cot}"
    )

    client = None
    hf_bundle = None
    if args.backend == "api":
        client = LLMClient(
            model_name=args.model,
            api_key=args.api_key,
            api_url=args.api_url,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
            enable_thinking=False,
        )
    else:
        hf_bundle = load_hf_model(args.hf_model_path, args.device)

    for ds_name, split_name, rows in splits:
        if ds_name not in wanted or split_name not in allowed_splits:
            continue
        if args.max_samples_per_split and args.max_samples_per_split > 0:
            rows = rows[: args.max_samples_per_split]

        if ds_name == "ToMBench":
            gen_fn = _v2_generate_fn(args, client, hf_bundle)
            tag = (
                f"tombench_v2_{args.backend}_sys{args.v2_language}_user{v2_user_lang}"
                f"_cot{int(args.v2_cot)}_votes{args.v2_shuffle_votes}"
            )
            summary = run_tombench_v2(
                rows,
                gen_fn,
                system_language=args.v2_language,
                user_language=v2_user_lang,
                cot=args.v2_cot,
                shuffle_votes=args.v2_shuffle_votes,
                base_seed=args.v2_seed,
                backend=args.backend,
                result_jsonl=result_dir / f"{tag}.jsonl",
                experiment_log=exp_log,
                log_steps=args.v2_log_steps,
            )
            (result_dir / f"{tag}_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            _log(f"[UNIT_DONE] ToMBench {tag} accuracy={summary['accuracy']:.6f} correct={summary['correct']}/{summary['total']}")

        elif ds_name == "Tomato":
            if args.tomato_mode == "mcqa":
                if args.backend != "hf":
                    raise RuntimeError("Tomato mcqa 目前仅支持 backend=hf（NLL 打分）")
                summary = run_tomato_v2(
                    rows,
                    backend="hf",
                    eval_mode="nll",
                    result_jsonl=result_dir / "tomato_v2_hf_nll.jsonl",
                    model=hf_bundle[0],
                    tokenizer=hf_bundle[1],
                    device=args.device,
                )
                sp = result_dir / "tomato_v2_hf_nll_summary.json"
            else:
                gen_fn = _v2_generate_fn(args, client, hf_bundle)
                if args.backend == "api":
                    summary = run_tomato_v2(
                        rows,
                        backend="api",
                        eval_mode="generate",
                        generate_fn=gen_fn,
                        result_jsonl=result_dir / "tomato_v2_api_generate.jsonl",
                    )
                    sp = result_dir / "tomato_v2_api_generate_summary.json"
                else:
                    summary = run_tomato_v2(
                        rows,
                        backend="hf",
                        eval_mode="generate",
                        generate_fn=gen_fn,
                        result_jsonl=result_dir / "tomato_v2_hf_generate.jsonl",
                    )
                    sp = result_dir / "tomato_v2_hf_generate_summary.json"
            sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _log(
                f"[UNIT_DONE] Tomato {summary.get('pipeline', '')} "
                f"accuracy={summary['accuracy']:.6f} correct={summary['correct']}/{summary['total']}"
            )

    _log("[DONE] run.py pipeline=v2 finished")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    _run_v2_pipeline(args, result_dir)
    return

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
    judge_open_datasets = set(args.judge_open_datasets or [])
    judge_runner: Optional[LLMClient] = None
    print(f"[INFO] Model: {model_tag} @ {args.api_url}", flush=True)
    print(f"[INFO] {len(templates)} template(s). prompt_style={args.prompt_style} "
          f"shuffle_repeats={args.shuffle_repeats}", flush=True)
    if args.batch_chunk_size > 0:
        print(f"[INFO] batch_chunk_size={args.batch_chunk_size}（分块生成并持续写 STEP 日志）", flush=True)
    else:
        print("[INFO] batch_chunk_size<=0（整批生成后再统一打分）", flush=True)
    if args.log_pred_tail > 0:
        print(
            f"[INFO] log_pred: 每条输出末 {args.log_pred_tail} 字符 -> {args.log_pred_where} "
            f"(experiment.log / 终端与 run.log 见 --log-pred-where)",
            flush=True,
        )

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

            n_shuffles = _should_shuffle(dataset_name, args.prompt_style, args.shuffle_repeats)
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

                def score_chunk(
                    sub_jobs: List[SampleJob],
                    sub_results: List[Tuple[List[Any], Any]],
                ) -> None:
                    nonlocal judge_runner
                    if dataset_name in MCQ_VOTE_DATASETS:
                        _score_voted_mcq_batch(
                            sub_jobs, sub_results, summary[key], all_records,
                            exp_log, dataset_name, split_name, tmpl_name,
                            log_pred_tail=args.log_pred_tail,
                            log_pred_where=args.log_pred_where,
                        )
                    elif dataset_name in judge_open_datasets:
                        if judge_runner is None:
                            if not args.judge_api_key:
                                raise RuntimeError(
                                    f"Dataset {dataset_name} requires --judge-api-key "
                                    "when the configured judge endpoint requires authentication."
                                )
                            judge_runner = LLMClient(
                                model_name=args.judge_model,
                                api_key=args.judge_api_key,
                                api_url=args.judge_api_url,
                                temperature=0.0,
                                max_tokens=1024,
                                top_p=1.0,
                                enable_thinking=False,
                            )
                            print(
                                f"[INFO] Judge enabled for {sorted(judge_open_datasets)} "
                                f"via {args.judge_model} @ {args.judge_api_url}",
                                flush=True,
                            )
                        _score_judged_open_batch(
                            sub_jobs, sub_results, summary[key], all_records,
                            exp_log, dataset_name, split_name, tmpl_name, judge_runner,
                            log_pred_tail=args.log_pred_tail,
                            log_pred_where=args.log_pred_where,
                        )
                    else:
                        _score_standard_batch(
                            sub_jobs, sub_results, summary[key], all_records,
                            exp_log, dataset_name, split_name, tmpl_name,
                            log_pred_tail=args.log_pred_tail,
                            log_pred_where=args.log_pred_where,
                        )

                total_jobs = len(jobs)
                exp_log.write(f"[BATCH] {tmpl_name} | {dataset_name}/{split_name} | {total_jobs} prompts")

                if args.batch_chunk_size > 0:
                    chunk_size = args.batch_chunk_size
                    for start in range(0, total_jobs, chunk_size):
                        end = min(start + chunk_size, total_jobs)
                        sub_jobs = jobs[start:end]
                        exp_log.write(
                            f"[BATCH_PART] {tmpl_name} | {dataset_name}/{split_name} | "
                            f"{start + 1}-{end}/{total_jobs}"
                        )
                        sub_results = runner.batch_generate([j.prompt for j in sub_jobs])
                        score_chunk(sub_jobs, sub_results)
                else:
                    batch_results = runner.batch_generate([j.prompt for j in jobs])
                    score_chunk(jobs, batch_results)

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
        results_table_path = result_dir / "results_table.md"
        if results_table_path.exists() and results_table_path.read_text(encoding="utf-8").strip():
            with results_table_path.open("a", encoding="utf-8") as f:
                f.write("\n\n---\n\n")
                f.write(table_doc)
        else:
            results_table_path.write_text(table_doc, encoding="utf-8")

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
