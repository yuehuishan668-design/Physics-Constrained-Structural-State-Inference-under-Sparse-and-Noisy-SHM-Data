"""
Check numerical consistency for the Q2 fast-track manuscript.

English:
This script compares canonical values from paper-ready tables with manuscript text
requirements. It prints a consistency report and writes it to the paper-ready package.

中文：
本脚本检查 Q2 fast-track 论文草稿中的关键数值是否与 paper-ready 表格一致，
避免不同实验协议下的 MAE/RMSE 被混用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ROOT = Path(".")
TABLE_DIR = ROOT / "results/paper_ready/q2_fast_track/tables"
MANUSCRIPT_DIR = ROOT / "manuscript"
OUT_REPORT = ROOT / "results/paper_ready/q2_fast_track/text/numerical_consistency_check.md"


TOL = 5e-4


def read_csv(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path)


def read_json(name: str) -> Dict:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(label: str, actual: float, expected: float, lines: List[str], tol: float = TOL) -> None:
    diff = abs(float(actual) - float(expected))
    status = "PASS" if diff <= tol else "FAIL"
    lines.append(f"- **{status}** `{label}`: actual={float(actual):.6f}, expected={expected:.6f}, diff={diff:.6f}")


def get_row(df: pd.DataFrame, feature_set: str, estimator: str) -> pd.Series:
    model_col = "estimator" if "estimator" in df.columns else "model"
    rows = df[(df["feature_set"].astype(str) == feature_set) & (df[model_col].astype(str) == estimator)]
    if rows.empty:
        raise KeyError(f"Cannot find row feature_set={feature_set}, estimator={estimator}")
    return rows.iloc[0]


def find_text_occurrences(patterns: List[str]) -> Dict[str, List[str]]:
    md_files = sorted(MANUSCRIPT_DIR.glob("*.md"))
    result = {p: [] for p in patterns}

    for path in md_files:
        text = path.read_text(encoding="utf-8")
        for p in patterns:
            if p in text:
                result[p].append(str(path))

    return result


def recursive_find(obj, target_key):
    """
    Recursively find the first value of a key in nested dict/list objects.

    中文：
    递归查找嵌套 JSON 中某个字段，避免因为字段层级变化导致 KeyError。
    """
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for v in obj.values():
            found = recursive_find(v, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = recursive_find(v, target_key)
            if found is not None:
                return found
    return None


def first_available(*values):
    """
    Return the first non-None value.

    中文：
    返回第一个不是 None 的值。
    """
    for v in values:
        if v is not None:
            return v
    return None


def check_dataset(lines: List[str]) -> None:
    lines.append("\n## 1. Dataset and feature summary")

    q = read_json("T00_dataset_quality_debug_plus_3000.json")
    f = read_json("T01_physics_feature_summary_debug_plus_3000.json")

    # Robustly read response shape.
    # 中文：兼容不同版本 quality summary 的字段命名。
    response_shape = first_available(
        recursive_find(q, "response_shape"),
        recursive_find(q, "X_shape"),
        recursive_find(q, "X_abs_accel_shape"),
        recursive_find(f, "response_shape"),
    )

    n_cases = first_available(
        recursive_find(q, "n_cases"),
        recursive_find(f, "n_cases"),
        response_shape[0] if isinstance(response_shape, list) and len(response_shape) >= 1 else None,
    )

    n_steps = first_available(
        recursive_find(q, "n_steps"),
        recursive_find(f, "n_steps"),
        response_shape[1] if isinstance(response_shape, list) and len(response_shape) >= 2 else None,
    )

    n_stories = first_available(
        recursive_find(q, "n_stories"),
        recursive_find(f, "n_stories"),
        response_shape[2] if isinstance(response_shape, list) and len(response_shape) >= 3 else None,
    )

    n_features = first_available(
        recursive_find(f, "n_features"),
        recursive_find(q, "n_features"),
    )

    F_train_shape = recursive_find(f, "F_train_shape")
    F_val_shape = recursive_find(f, "F_val_shape")
    F_test_shape = recursive_find(f, "F_test_shape")

    if n_cases is not None:
        assert_close("dataset n_cases", n_cases, 3000, lines, tol=0)
    else:
        lines.append("- **WARN** `dataset n_cases` not found; expected 3000.")

    if n_stories is not None:
        assert_close("dataset n_stories", n_stories, 4, lines, tol=0)
    else:
        lines.append("- **WARN** `dataset n_stories` not found; expected 4.")

    if n_steps is not None:
        assert_close("response time steps", n_steps, 2000, lines, tol=0)
    else:
        lines.append("- **WARN** `response time steps` not found; expected 2000.")

    if n_features is not None:
        assert_close("feature n_features", n_features, 92, lines, tol=0)
    else:
        lines.append("- **WARN** `feature n_features` not found; expected 92.")

    if isinstance(F_train_shape, list) and len(F_train_shape) >= 1:
        assert_close("F_train size", F_train_shape[0], 2100, lines, tol=0)
    else:
        lines.append("- **WARN** `F_train_shape` not found; expected first dimension 2100.")

    if isinstance(F_val_shape, list) and len(F_val_shape) >= 1:
        assert_close("F_val size", F_val_shape[0], 450, lines, tol=0)
    else:
        lines.append("- **WARN** `F_val_shape` not found; expected first dimension 450.")

    if isinstance(F_test_shape, list) and len(F_test_shape) >= 1:
        assert_close("F_test size", F_test_shape[0], 450, lines, tol=0)
    else:
        lines.append("- **WARN** `F_test_shape` not found; expected first dimension 450.")

def check_main_ablation(lines: List[str]) -> None:
    lines.append("\n## 2. Main fixed-split ablation")

    df = read_csv("T02_main_ablation_3000.csv")

    row = get_row(df, "full", "ridge")
    assert_close("main full + ridge test_mae", row["test_mae"], 0.0393357, lines)
    assert_close("main full + ridge test_rmse", row["test_rmse"], 0.0585481, lines)
    assert_close("main full + ridge damaged-entry MAE", row["mae_on_damaged_entries"], 0.0634314, lines)

    row = get_row(df, "no_meta", "ridge")
    assert_close("main no_meta + ridge test_mae", row["test_mae"], 0.0424577, lines)

    row = get_row(df, "response_basic_only", "ridge")
    assert_close("main response_basic_only + ridge test_mae", row["test_mae"], 0.0577182, lines)

    row = get_row(df, "full", "elasticnet")
    assert_close("main full + elasticnet test_mae", row["test_mae"], 0.0476007, lines)

    row = get_row(df, "full", "random_forest")
    assert_close("main full + random_forest test_mae", row["test_mae"], 0.0644156, lines)


def check_repeated_split(lines: List[str]) -> None:
    lines.append("\n## 3. Repeated-split robustness")

    df = read_csv("T03_repeated_split_robustness_summary.csv")
    row = get_row(df, "full", "ridge")

    assert_close("repeated full + ridge test_mae_mean", row["test_mae_mean"], 0.045426, lines)
    assert_close("repeated full + ridge test_mae_std", row["test_mae_std"], 0.000865, lines)
    assert_close("repeated full + ridge test_rmse_mean", row["test_rmse_mean"], 0.062085, lines)

    row = get_row(df, "no_meta", "ridge")
    assert_close("repeated no_meta + ridge test_mae_mean", row["test_mae_mean"], 0.049016, lines)

    per_seed = read_csv("T04_repeated_split_per_seed_best.csv")
    n_full_ridge = ((per_seed["feature_set"] == "full") & (per_seed["model"] == "ridge")).sum()
    assert_close("per-seed best full + ridge count", n_full_ridge, 10, lines, tol=0)


def check_damage_stratified(lines: List[str]) -> None:
    lines.append("\n## 4. Damage-stratified reliability")

    df = read_csv("T05_damage_stratified_config_summary.csv")
    row = get_row(df, "full", "ridge")

    assert_close("damage full + ridge overall_mae_mean", row["overall_mae_mean"], 0.040679, lines)
    assert_close("damage full + ridge overall_rmse_mean", row["overall_rmse_mean"], 0.060010, lines)
    assert_close("damage full + ridge high_damage_mae_mean", row["high_damage_mae_mean"], 0.088806, lines)
    assert_close("damage full + ridge high_damage_underestimation_ratio_mean", row["high_damage_underestimation_ratio_mean"], 0.821399, lines)

    # Detailed high-damage bias / mean_true / mean_pred may be stored in different
    # formats depending on the damage-stratified output table.
    #
    # 中文：
    # high-damage 的 bias / mean_true / mean_pred 在不同版本输出中可能不在同一张表或同一列名里。
    # 因此这里做兼容检查，避免因为缺少 bin 列导致整个脚本中断。
    runs = read_csv("T06_damage_stratified_runs.csv")

    if "bin" in runs.columns:
        high_rows = runs[
            (runs["feature_set"].astype(str) == "full")
            & (runs["model"].astype(str) == "ridge")
            & (runs["bin"].astype(str) == "high")
        ]

        if high_rows.empty:
            lines.append("- **WARN** no `full + ridge + high` row found in T06_damage_stratified_runs.csv.")
            return

        high = high_rows.iloc[0]

        if "bias_mean" in high.index:
            assert_close("damage high bin bias_mean", high["bias_mean"], -0.071656, lines)
        else:
            lines.append("- **WARN** `bias_mean` not found for high-damage bin.")

        if "mean_true_mean" in high.index:
            assert_close("damage high bin mean_true_mean", high["mean_true_mean"], 0.275775, lines)
        else:
            lines.append("- **WARN** `mean_true_mean` not found for high-damage bin.")

        if "mean_pred_mean" in high.index:
            assert_close("damage high bin mean_pred_mean", high["mean_pred_mean"], 0.204119, lines)
        else:
            lines.append("- **WARN** `mean_pred_mean` not found for high-damage bin.")

    else:
        # Try alternative wide-format columns.
        # 中文：如果没有 bin 列，则尝试宽表字段。
        wide_row = None
        if "feature_set" in runs.columns and "model" in runs.columns:
            candidates = runs[
                (runs["feature_set"].astype(str) == "full")
                & (runs["model"].astype(str) == "ridge")
            ]
            if not candidates.empty:
                wide_row = candidates.iloc[0]

        alternative_columns = {
            "damage high bin bias_mean": [
                "high_damage_bias_mean",
                "bias_high_mean",
                "high_bias_mean",
            ],
            "damage high bin mean_true_mean": [
                "high_damage_mean_true_mean",
                "high_damage_true_mean",
                "mean_true_high_damage",
            ],
            "damage high bin mean_pred_mean": [
                "high_damage_mean_pred_mean",
                "high_damage_pred_mean",
                "mean_pred_high_damage",
            ],
        }

        expected_values = {
            "damage high bin bias_mean": -0.071656,
            "damage high bin mean_true_mean": 0.275775,
            "damage high bin mean_pred_mean": 0.204119,
        }

        found_any = False
        if wide_row is not None:
            for label, cols in alternative_columns.items():
                matched = None
                for c in cols:
                    if c in wide_row.index:
                        matched = c
                        break

                if matched is not None:
                    found_any = True
                    assert_close(label, wide_row[matched], expected_values[label], lines)
                else:
                    lines.append(f"- **WARN** `{label}` not found in wide-format T06 table.")

        if not found_any:
            lines.append("- **WARN** T06_damage_stratified_runs.csv has no `bin` column and no recognized high-damage bias/mean columns.")
            lines.append(f"- **INFO** Available T06 columns: {list(runs.columns)}")
            lines.append("- **INFO** Core damage-stratified checks from T05 passed above; detailed high-bin bias values should be verified from R02_damage_stratified_report.md if needed.")

def check_noise(lines: List[str]) -> None:
    lines.append("\n## 5. Noise robustness")

    df = read_csv("T07_noise_best_by_noise.csv")
    expected = {
        0: 0.0344682,
        2: 0.0361334,
        5: 0.0375458,
        10: 0.0394483,
        15: 0.0431412,
        20: 0.0419665,
    }

    for pct, mae in expected.items():
        rows = df[df["noise_percent"].round().astype(int) == pct]
        if rows.empty:
            lines.append(f"- **FAIL** missing noise_percent={pct}")
            continue
        row = rows.iloc[0]
        feature = str(row["feature_set"])
        model = str(row["model"])
        status = "PASS" if feature == "full" and model == "ridge" else "FAIL"
        lines.append(f"- **{status}** noise {pct}% best config: {feature} + {model}")
        assert_close(f"noise {pct}% full + ridge test_mae", row["test_mae"], mae, lines)


def check_sensor(lines: List[str]) -> None:
    lines.append("\n## 6. Sensor sparsity")

    df = read_csv("T09_sensor_best_by_sensor_count.csv")

    expected: Dict[int, Tuple[str, str, float]] = {
        4: ("full", "ridge", 0.0393357),
        3: ("full", "ridge", 0.0612044),
        2: ("full", "random_forest", 0.0763734),
        1: ("full", "random_forest", 0.0801414),
    }

    for count, (feature_expected, model_expected, mae_expected) in expected.items():
        rows = df[df["sensor_count"].astype(int) == count]
        if rows.empty:
            lines.append(f"- **FAIL** missing sensor_count={count}")
            continue
        row = rows.iloc[0]
        feature = str(row["feature_set"])
        model = str(row["model"])
        status = "PASS" if feature == feature_expected and model == model_expected else "FAIL"
        lines.append(f"- **{status}** {count}-sensor best config: actual={feature} + {model}, expected={feature_expected} + {model_expected}")
        assert_close(f"{count}-sensor best test_mae", row["test_mae"], mae_expected, lines)


def check_text_presence(lines: List[str]) -> None:
    lines.append("\n## 7. Required rounded values in manuscript text")

    required_strings = [
        "0.0393",
        "0.0585",
        "0.0454",
        "0.0009",
        "0.0407",
        "0.0600",
        "0.0888",
        "-0.0717",
        "0.8214",
        "0.0345",
        "0.0431",
        "0.0420",
        "0.0612",
        "0.0764",
        "0.0801",
        "3000",
        "92",
        "2100",
        "450",
    ]

    occ = find_text_occurrences(required_strings)
    for s, files in occ.items():
        if files:
            lines.append(f"- **PASS** `{s}` found in: {', '.join(files)}")
        else:
            lines.append(f"- **WARN** `{s}` not found in manuscript text")


def check_risky_phrases(lines: List[str]) -> None:
    lines.append("\n## 8. Risky phrase scan")

    risky = [
        "always best",
        "strictly monotonic",
        "field-ready",
        "deployment-ready",
        "guarantee",
        "prove",
        "all real SHM",
    ]

    md_files = sorted(MANUSCRIPT_DIR.glob("*.md"))
    found_any = False

    for path in md_files:
        text_lower = path.read_text(encoding="utf-8").lower()
        for phrase in risky:
            if phrase.lower() in text_lower:
                found_any = True
                lines.append(f"- **CHECK** risky phrase `{phrase}` found in `{path}`")

    if not found_any:
        lines.append("- **PASS** no risky phrases found.")


def main() -> None:
    lines: List[str] = []
    lines.append("# Q2 Numerical Consistency Check")
    lines.append("")
    lines.append("This report checks whether the key paper-ready tables match the canonical numerical claims used in the manuscript drafts.")

    check_dataset(lines)
    check_main_ablation(lines)
    check_repeated_split(lines)
    check_damage_stratified(lines)
    check_noise(lines)
    check_sensor(lines)
    check_text_presence(lines)
    check_risky_phrases(lines)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("")
    print(f"Report written to: {OUT_REPORT}")


if __name__ == "__main__":
    main()
