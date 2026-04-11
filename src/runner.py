"""数据集评测运行器公共函数

提供数据集评测脚本之间的共享逻辑，减少重复代码。
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml

from src.dataloader import load_dataset
from src.llm import LLMClient
import logging
# 将日志级别设置为 WARNING 或更高
logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """加载数据集配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        数据集配置字典
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

        # 动态导入数据集的 schemas 模块
        dataset_dir = Path(config_path).parent
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"{config['dataset']}.schemas",
            dataset_dir / "schemas.py"
        )
        schemas_module = importlib.util.module_from_spec(spec)
        sys.modules[f"{config['dataset']}.schemas"] = schemas_module
        spec.loader.exec_module(schemas_module)

        # 从 SCHEMAS 字典获取 schema
        schema_name = config["schema"]
        schema = schemas_module.SCHEMAS[schema_name]

        return {
            "dataset": config["dataset"],
            "subset": config["path"],
            "schema": schema,
            "default_prompt": config["default_prompt"],
            "schemas_module": schemas_module,  # 暴露给 run.py 用于内部调用其他 schema
        }


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """加载实验配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        实验配置字典（llm_config, repeats, max_samples, datasets_path, results_path, judge_config 等）
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        return {
            "llm_config": config.get("llm", {}),
            "repeats": config.get("repeats", 1),
            "max_samples": config.get("max_samples", 0),
            "datasets_path": config.get("datasets_path", "datasets"),
            "results_path": config.get("results_path", "results"),
            "judge_config": config.get("judge", {}),  # 覆盖数据集的 judge 配置
        }


def create_llm_client(llm_config: Dict[str, Any]) -> LLMClient:
    """创建 LLM 客户端

    Args:
        llm_config: 配置字典，包含 model_name, api_key, api_url, temperature, max_tokens 等

    Returns:
        LLMClient 实例
    """
    config = llm_config.copy()

    return LLMClient.from_config(config)


def _compute_average_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算所有 metrics 的平均值

    Args:
        all_metrics: 所有重复运行的 metrics 列表

    Returns:
        平均 metrics 字典
    """
    if not all_metrics:
        return {}

    avg_metrics = {}

    for key in all_metrics[0].keys():
        values = [m.get(key) for m in all_metrics if key in m]
        if not values:
            continue

        first_value = values[0]
        if isinstance(first_value, (int, float)):
            # 数值类型直接计算平均
            avg_metrics[key] = sum(v for v in values if isinstance(v, (int, float))) / len(values)
        elif isinstance(first_value, dict):
            # 字典类型递归计算平均
            sub_avg = {}
            sub_keys = set()
            for v in values:
                if isinstance(v, dict):
                    sub_keys.update(v.keys())

            for sub_key in sub_keys:
                sub_values = [v.get(sub_key) for v in values if isinstance(v, dict) and sub_key in v]
                if sub_values and isinstance(sub_values[0], (int, float)):
                    sub_avg[sub_key] = sum(sub_values) / len(sub_values)

            avg_metrics[key] = sub_avg

    return avg_metrics


def save_common_results(
    dataset_name: str,
    model: str,
    prompt_method: str,
    all_predictions: List[List[str]],
<<<<<<< HEAD
    gold_answers,
=======
    gold_answers: List[str],
>>>>>>> upstream/main
    all_metrics: List[Dict[str, Any]],
    results_path: str = "results",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """保存评测结果

    Args:
        dataset_name: 数据集名称
        model: 模型名称
        prompt_method: prompt 方法
        all_predictions: 所有重复运行的预测结果列表 [repeat][sample]
<<<<<<< HEAD
        gold_answers: 标准答案，支持两种格式：
            - List[str]: 所有 repeat 共用同一组 gold（如 ToMBench）
            - List[List[str]]: 每个 repeat 有独立 gold（如 Tomato shuffle）
=======
        gold_answers: 标准答案列表
>>>>>>> upstream/main
        all_metrics: 所有重复运行的 metrics 列表
        results_path: 结果保存路径
        metadata: 额外元数据（如 judge_model）

    Returns:
        (jsonl_path, json_path) 元组
    """
    results_dir = Path(results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset_name}_{model}_{prompt_method}_{timestamp}"

<<<<<<< HEAD
    per_repeat_gold = bool(gold_answers and isinstance(gold_answers[0], list))

=======
>>>>>>> upstream/main
    # 1. 保存 predictions 到 jsonl
    jsonl_path = results_dir / f"{base_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for repeat_idx, predictions in enumerate(all_predictions):
<<<<<<< HEAD
            repeat_gold = gold_answers[repeat_idx] if per_repeat_gold else gold_answers
            for sample_idx, (pred, gold) in enumerate(zip(predictions, repeat_gold)):
=======
            for sample_idx, (pred, gold) in enumerate(zip(predictions, gold_answers)):
>>>>>>> upstream/main
                record = {
                    "repeat": repeat_idx,
                    "sample_idx": sample_idx,
                    "prediction": pred,
                    "gold_answer": gold,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. 计算 average metrics 并保存汇总指标到 json
    avg_metrics = _compute_average_metrics(all_metrics)

    json_path = results_dir / f"{base_name}.json"
    json_data = {
        "dataset": dataset_name,
        "model": model,
        "prompt_method": prompt_method,
        "repeats": len(all_metrics),
        "avg_metrics": avg_metrics,
        "all_metrics": all_metrics,
        "timestamp": timestamp,
    }
    if metadata:
        json_data["metadata"] = metadata

    json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Results saved to: {jsonl_path} and {json_path}")
    return jsonl_path, json_path


def print_summary_stats(
    all_metrics: List[Dict[str, Any]],
    repeats: int,
    total_samples: int,
) -> None:
    """打印统计摘要

    Args:
        all_metrics: 所有重复运行的 metrics 列表
        repeats: 重复次数
        total_samples: 总样本数
    """
    avg_accuracy = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
    avg_correct = sum(m["correct"] for m in all_metrics) / len(all_metrics)

    print(f"\n{'='*50}")
    print(f"Results (Average over {repeats} runs)")
    print(f"{'='*50}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Correct: {avg_correct:.1f}/{total_samples}")
    print(f"{'='*50}")


def load_and_limit_data(
    subset: str,
    datasets_path: str = "datasets",
    max_samples: int = 0,
) -> List[Dict[str, Any]]:
    """加载数据并限制样本数

    Args:
        subset: 数据集子集路径
        datasets_path: 数据集根目录
        max_samples: 最大样本数（0 表示不限制）

    Returns:
        数据列表
    """
    data = load_dataset(subset, datasets_root=datasets_path)
    if max_samples > 0:
        data = data[:max_samples]
    return data
