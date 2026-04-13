"""ToMi 评测脚本（单词答案版）"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner

from ToMi.prompts import get_template, build_prompt
from ToMi.metrics import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_tomi_json(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """从 JSON 文件加载 ToMi 数据。"""
    data_path = Path(path)
    if not data_path.is_absolute():
        # 兼容从项目根目录外启动脚本的场景
        project_relative = PROJECT_ROOT / data_path
        if project_relative.exists():
            data_path = project_relative

    if not data_path.exists():
        raise FileNotFoundError(
            f"ToMi data file not found: {data_path}. "
            "Please update ToMi/config.yaml:path to the real JSON location."
        )

    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {data_path}, got {type(data).__name__}")

    if max_samples > 0:
        data = data[:max_samples]
    return data


def extract_gold_answers(data: List[Dict[str, Any]]) -> List[str]:
    """提取标准答案。"""
    return [str(row.get("output", "")).strip().lower() for row in data]


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config(str(PROJECT_ROOT / "ToMi/config.yaml"))

    # 加载实验配置
    experiment_config = runner.load_experiment_config(str(PROJECT_ROOT / "experiment_config.yaml"))

    schema = dataset_config["schema"]
    prompt_method = dataset_config["default_prompt"]

    # 获取 prompt 模板
    template = get_template(prompt_method)

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"])

    # 加载数据（ToMi 使用 JSON 文件）
    data = load_tomi_json(
        path=dataset_config["subset"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['subset']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    # 构建 prompts（每个 repeat 构建相同的 prompts）
    prompts = [build_prompt(template, row) for row in data]
    all_prompts = prompts * experiment_config["repeats"]

    # 批量结构化推理
    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    # 计算 metrics
    all_predictions = []
    all_metrics = []
    for i in range(experiment_config["repeats"]):
        start = i * len(data)
        end = start + len(data)
        repeat_results = results[start:end]
        predictions = [r.answer for r in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    # 保存结果
    gold_answers = extract_gold_answers(data)
    runner.save_common_results(
        dataset_name=dataset_config["dataset"],
        model=experiment_config["llm_config"]["model_name"],
        prompt_method=prompt_method,
        all_predictions=all_predictions,
        gold_answers=gold_answers,
        all_metrics=all_metrics,
        results_path=experiment_config["results_path"],
        dataset_config=dataset_config,
        experiment_config=experiment_config,
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(gold_answers))


if __name__ == "__main__":
    main()
