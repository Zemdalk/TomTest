"""统一评测入口

运行所有数据集的评测脚本。
所有配置通过 experiment_config.yaml 管理，无需命令行参数。
"""
import subprocess
import sys
from pathlib import Path


# 数据集列表
DATASETS = [
    "ToMBench",
<<<<<<< HEAD
    "Tomato",
=======
>>>>>>> upstream/main
]


def run_dataset(dataset: str) -> bool:
    """运行指定数据集的评测

    Args:
        dataset: 数据集名称

    Returns:
        是否成功
    """
    run_script = Path(f"{dataset}/run.py")
    if not run_script.exists():
        print(f"[{dataset}] run.py not found, skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {dataset}")
    print(f"{'='*60}")

    try:
        subprocess.run([sys.executable, str(run_script)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{dataset}] Error: {e}")
        return False


def main():
    for dataset in DATASETS:
        run_dataset(dataset)

    print(f"\n{'='*60}")
    print("All datasets completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()