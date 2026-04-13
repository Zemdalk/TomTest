"""ToMQA 评测脚本（基于结构化输出）"""
import sys
from pathlib import Path

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner
from ToMQA.prompts import get_template, build_prompt
from ToMQA.metrics import compute_metrics

import logging

# 关闭不必要日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def extract_gold_answers(data):
    """提取标准答案（取 Correct_Answer 列表的第一个）"""
    golds = []
    for row in data:
        answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
        correct_list = answer_block.get("Correct_Answer", [])
        if isinstance(correct_list, list) and correct_list:
            golds.append(correct_list[0])
        elif isinstance(correct_list, list):
            golds.append("")
        elif correct_list is None:
            golds.append("")
        else:
            golds.append(str(correct_list))
    return golds


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("ToMQA/config.yaml")

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

    print(f"Loaded {len(data)} samples from {dataset_config['subset']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Repeats: {experiment_config['repeats']}")

    # 构建 prompts
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
        predictions = [getattr(r, "answer", "") for r in repeat_results]
        all_predictions.append(predictions)

        metrics = compute_metrics(predictions, data)
        all_metrics.append(metrics)
        print(
            f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, "
            f"Correct={metrics['correct']}/{metrics['total']}"
        )

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
