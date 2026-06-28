"""
Check dataset quality before large-scale model training.

English:
This script inspects a raw SHM dataset and, optionally, a normalized split dataset.
It is a quality gate before moving from debug_plus_100 to debug_plus_500 / debug_plus_1000.

中文：
本脚本用于检查原始结构响应数据集，以及可选的训练/验证/测试划分数据集。
它是从 debug_plus_100 扩展到 debug_plus_500 / debug_plus_1000 前的质量检查步骤。

Typical command:
python -m src.evaluation.check_damage_dataset_quality \
  --raw-dataset data_processed/debug_plus_500_dataset.npz \
  --split-dataset data_processed/debug_plus_500_split_normalized.npz \
  --output-json results/tables/dataset_quality/debug_plus_500_quality_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def to_float(x: Any) -> float:
    """
    English:
    Convert numpy scalar to Python float.

    中文：
    将 numpy 标量转换为 Python float，方便保存 JSON。
    """
    return float(np.asarray(x).item())


def to_int(x: Any) -> int:
    """
    English:
    Convert numpy scalar to Python int.

    中文：
    将 numpy 标量转换为 Python int，方便保存 JSON。
    """
    return int(np.asarray(x).item())


def quantile_dict(x: np.ndarray) -> Dict[str, float]:
    """
    English:
    Compute common quantiles.

    中文：
    计算常用分位数，用于观察数据分布是否异常。
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return {}
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    return {str(q): to_float(np.quantile(x, q)) for q in qs}


def find_first_key(data: np.lib.npyio.NpzFile, candidates: List[str]) -> Optional[str]:
    """
    English:
    Find the first existing key in an npz file.

    中文：
    在 npz 文件中寻找第一个存在的候选 key。
    """
    keys = set(data.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def summarize_damage(y: np.ndarray) -> Dict[str, Any]:
    """
    English:
    Summarize damage labels.

    中文：
    汇总损伤标签分布。
    """
    y = np.asarray(y, dtype=float)
    damage_sum = np.sum(y, axis=1)
    damage_max = np.max(y, axis=1)
    nonzero_count = np.sum(y > 1e-12, axis=1)

    return {
        "n_cases": to_int(y.shape[0]),
        "n_stories": to_int(y.shape[1]) if y.ndim == 2 else "unknown",
        "healthy_case_count": to_int(np.sum(damage_sum <= 1e-12)),
        "healthy_case_ratio": to_float(np.mean(damage_sum <= 1e-12)),
        "damaged_case_count": to_int(np.sum(damage_sum > 1e-12)),
        "severe_case_count_damage_max_ge_0p20": to_int(np.sum(damage_max >= 0.20)),
        "severe_case_ratio_damage_max_ge_0p20": to_float(np.mean(damage_max >= 0.20)),
        "damage_sum_quantiles": quantile_dict(damage_sum),
        "damage_max_quantiles": quantile_dict(damage_max),
        "damaged_story_count_distribution": dict(Counter(nonzero_count.astype(int).tolist())),
        "story_damage_frequency": np.mean(y > 1e-12, axis=0).astype(float).tolist(),
        "story_damage_mean": np.mean(y, axis=0).astype(float).tolist(),
        "story_damage_max": np.max(y, axis=0).astype(float).tolist(),
    }


def summarize_response(X: np.ndarray) -> Dict[str, Any]:
    """
    English:
    Summarize response amplitude.

    中文：
    汇总响应幅值分布，主要检查是否存在极端异常响应样本。
    """
    X = np.asarray(X, dtype=float)
    max_per_case = np.max(np.abs(X), axis=tuple(range(1, X.ndim)))
    rms_per_case = np.sqrt(np.mean(X**2, axis=tuple(range(1, X.ndim))))

    return {
        "response_shape": list(X.shape),
        "max_abs_response_quantiles": quantile_dict(max_per_case),
        "rms_response_quantiles": quantile_dict(rms_per_case),
        "global_min": to_float(np.min(X)),
        "global_max": to_float(np.max(X)),
    }


def summarize_optional_vector(data: np.lib.npyio.NpzFile, key: str) -> Optional[Dict[str, Any]]:
    """
    English:
    Summarize an optional metadata vector if it exists.

    中文：
    如果某个元数据向量存在，则汇总其分布。
    """
    if key not in data.files:
        return None

    x = np.asarray(data[key])
    if x.ndim == 0:
        return {"value": to_float(x)}

    if np.issubdtype(x.dtype, np.number):
        return {
            "count": to_int(x.size),
            "min": to_float(np.min(x)),
            "max": to_float(np.max(x)),
            "mean": to_float(np.mean(x)),
            "quantiles": quantile_dict(x),
            "value_counts": dict(Counter(np.round(x.astype(float), 6).astype(str).tolist())),
        }

    return {"count": to_int(x.size), "value_counts": dict(Counter(x.astype(str).tolist()))}


def summarize_raw_dataset(path: Path) -> Dict[str, Any]:
    """
    English:
    Summarize the raw dataset.

    中文：
    汇总原始数据集。
    """
    data = np.load(path, allow_pickle=True)

    response_key = find_first_key(data, ["X_abs_accel", "X", "response", "responses"])
    damage_key = find_first_key(data, ["y_damage", "y", "damage", "labels"])

    if response_key is None:
        raise KeyError(f"No response key found in {path}. Available keys: {data.files}")
    if damage_key is None:
        raise KeyError(f"No damage label key found in {path}. Available keys: {data.files}")

    X = np.asarray(data[response_key])
    y = np.asarray(data[damage_key])

    summary: Dict[str, Any] = {
        "path": str(path),
        "available_keys": list(data.files),
        "response_key": response_key,
        "damage_key": damage_key,
        "damage": summarize_damage(y),
        "response": summarize_response(X),
        "metadata": {},
    }

    for key in ["amplitude_g", "frequency_hz", "noise_level", "case_id"]:
        value = summarize_optional_vector(data, key)
        if value is not None:
            summary["metadata"][key] = value

    return summary


def summarize_split_dataset(path: Path) -> Dict[str, Any]:
    """
    English:
    Summarize train/val/test split dataset.

    中文：
    汇总训练集、验证集、测试集划分后的数据。
    """
    data = np.load(path, allow_pickle=True)
    summary: Dict[str, Any] = {
        "path": str(path),
        "available_keys": list(data.files),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        x_key = f"X_{split}"
        y_key = f"y_{split}"
        if x_key not in data.files or y_key not in data.files:
            continue

        X = np.asarray(data[x_key])
        y = np.asarray(data[y_key])
        summary["splits"][split] = {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "damage": summarize_damage(y),
            "response": summarize_response(X),
        }

        for meta_base in ["amplitude_g", "frequency_hz", "noise_level", "case_id"]:
            meta_key = f"{meta_base}_{split}"
            value = summarize_optional_vector(data, meta_key)
            if value is not None:
                summary["splits"][split].setdefault("metadata", {})[meta_key] = value

    return summary


def add_quality_flags(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    English:
    Add simple quality flags. These are not absolute scientific criteria; they are early warnings.

    中文：
    添加简单质量警戒项。这些不是绝对科研标准，只是早期排错用。
    """
    raw_damage = summary["raw"]["damage"]
    n_cases = raw_damage["n_cases"]
    healthy_ratio = raw_damage["healthy_case_ratio"]
    severe_ratio = raw_damage["severe_case_ratio_damage_max_ge_0p20"]

    flags = []

    if n_cases < 500:
        flags.append("n_cases_lt_500: dataset is still too small for stable model comparison.")
    if healthy_ratio < 0.10:
        flags.append("healthy_ratio_lt_0p10: too few healthy cases.")
    if healthy_ratio > 0.40:
        flags.append("healthy_ratio_gt_0p40: too many healthy cases; damaged cases may be insufficient.")
    if severe_ratio < 0.10:
        flags.append("severe_damage_ratio_lt_0p10: too few high-damage cases; models may underpredict large damage.")

    split_summary = summary.get("split", {}).get("splits", {})
    for split_name, info in split_summary.items():
        split_healthy = info["damage"]["healthy_case_ratio"]
        if split_healthy < 0.05 or split_healthy > 0.50:
            flags.append(f"{split_name}_healthy_ratio_unbalanced: {split_healthy:.3f}")

    summary["quality_flags"] = flags
    summary["quality_status"] = "pass" if not flags else "warning"
    return summary


def parse_args() -> argparse.Namespace:
    """
    English:
    Parse command-line arguments.

    中文：
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Check SHM damage dataset quality before large-scale training.")
    parser.add_argument("--raw-dataset", type=Path, required=True, help="Raw dataset npz path.")
    parser.add_argument("--split-dataset", type=Path, default=None, help="Optional split/normalized dataset npz path.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    """
    English:
    Main execution function.

    中文：
    主执行函数。
    """
    args = parse_args()

    summary: Dict[str, Any] = {
        "raw": summarize_raw_dataset(args.raw_dataset),
    }

    if args.split_dataset is not None:
        summary["split"] = summarize_split_dataset(args.split_dataset)

    summary = add_quality_flags(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    raw_damage = summary["raw"]["damage"]
    raw_response = summary["raw"]["response"]

    print("Dataset quality check completed.")
    print(f"Raw dataset: {args.raw_dataset}")
    print(f"n_cases: {raw_damage['n_cases']}")
    print(f"healthy_case_ratio: {raw_damage['healthy_case_ratio']:.6f}")
    print(f"severe_case_ratio_damage_max_ge_0p20: {raw_damage['severe_case_ratio_damage_max_ge_0p20']:.6f}")
    print(f"damage_max_quantiles: {raw_damage['damage_max_quantiles']}")
    print(f"max_abs_response_quantiles: {raw_response['max_abs_response_quantiles']}")
    print(f"quality_status: {summary['quality_status']}")

    if summary["quality_flags"]:
        print("quality_flags:")
        for flag in summary["quality_flags"]:
            print(f"- {flag}")

    if args.output_json is not None:
        print(f"Summary JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
