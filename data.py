"""数据加载与 schema 规范化。"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_from_disk


@dataclass
class SampleMeta:
    dataset: str
    split: str
    index: int


def load_dataset_splits(dataset_root: Path) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    """返回 [(dataset_name, split_name, rows), ...]。"""
    results = []
    for ds_dir in sorted(dataset_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name == ".cache":
            continue
        if (ds_dir / "dataset_dict.json").exists():
            # DatasetDict 格式（ToMi、ToMQA）
            ds = load_from_disk(str(ds_dir))
            for split_name, split_ds in ds.items():
                results.append((ds_dir.name, split_name, list(split_ds)))
        else:
            # 逐 split 子目录（Tomato、ToMBench）
            for split_dir in sorted(ds_dir.iterdir()):
                if split_dir.is_dir() and (split_dir / "dataset_info.json").exists():
                    results.append((ds_dir.name, split_dir.name, list(load_from_disk(str(split_dir)))))
    return results


def to_json_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False)
    return str(obj)


def extract_answers(answer_obj: Any) -> List[str]:
    if isinstance(answer_obj, dict):
        cands = answer_obj.get("Correct_Answer") or answer_obj.get("Correct Answer", [])
        if isinstance(cands, list):
            return [str(x) for x in cands if str(x).strip()]
        if isinstance(cands, str) and cands.strip():
            return [cands]
    if isinstance(answer_obj, list):
        return [str(x) for x in answer_obj if str(x).strip()]
    if isinstance(answer_obj, str) and answer_obj.strip():
        return [answer_obj]
    return []


def extract_wrong_answers(answer_obj: Any) -> List[str]:
    if isinstance(answer_obj, dict):
        w = answer_obj.get("Wrong_Answer") or answer_obj.get("Wrong Answer", [])
        if isinstance(w, list):
            return [str(x) for x in w if str(x).strip()]
        if isinstance(w, str) and w.strip():
            return [w]
    return []


def normalize_tom_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """统一各数据集的字段命名差异。"""
    out = dict(row)
    if isinstance(out.get("Story"), str):
        out["Story"] = {"full_story": out["Story"].strip(), "summary": "", "background": []}
    if isinstance(out.get("State"), dict):
        st = dict(out["State"])
        if "Human State" in st and "Human_State" not in st:
            st["Human_State"] = st.pop("Human State")
        if "Environment State" in st and "Environment_State" not in st:
            st["Environment_State"] = st.pop("Environment State")
        out["State"] = st
    if isinstance(out.get("Answer"), dict):
        ans = dict(out["Answer"])
        if "Correct_Answer" not in ans and "Correct Answer" in ans:
            ans["Correct_Answer"] = ans["Correct Answer"]
        if "Wrong_Answer" not in ans and "Wrong Answer" in ans:
            ans["Wrong_Answer"] = ans["Wrong Answer"]
        out["Answer"] = ans
    return out
