#!/usr/bin/env python3
"""Verify frozen SCHM manuscript anchors without training or retuning."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAILURES: list[str] = []


def rows(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(data: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    selected = [
        row for row in data
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(selected) != 1:
        raise AssertionError(f"Expected one row for {criteria}, found {len(selected)}")
    return selected[0]


def close(name: str, actual: float, expected: float, atol: float = 5e-10) -> None:
    if math.isclose(actual, expected, rel_tol=1e-9, abs_tol=atol):
        print(f"[PASS] {name}: {actual:.12g}")
    else:
        FAILURES.append(f"{name}: actual={actual!r}, expected={expected!r}")
        print(f"[FAIL] {name}: actual={actual!r}, expected={expected!r}")


def equal(name: str, actual: object, expected: object) -> None:
    if actual == expected:
        print(f"[PASS] {name}: {actual}")
    else:
        FAILURES.append(f"{name}: actual={actual!r}, expected={expected!r}")
        print(f"[FAIL] {name}: actual={actual!r}, expected={expected!r}")


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    print("SCHM FROZEN MANUSCRIPT ANCHOR VERIFICATION")
    print("Training performed: False")
    print("Hyperparameter tuning performed: False")
    print()

    factorial = rows(
        "results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/"
        "repeated_factorial_mean_table.csv"
    )
    p1_mae = one(factorial, metric="test_mae")
    close("P1 78D Ridge MAE", float(p1_mae["78D_Ridge"]), 0.04485829455702703)
    close("P1 78D RBF-SVR MAE", float(p1_mae["78D_SVR"]), 0.031450000610448225)
    close("P1 92D Ridge MAE", float(p1_mae["92D_Ridge"]), 0.03954549435827245)
    close("P1 92D RBF-SVR MAE", float(p1_mae["92D_SVR"]), 0.02310484788409254)

    effects = rows(
        "results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/"
        "repeated_factorial_effect_summary.csv"
    )
    interaction = one(
        effects,
        metric="test_mae",
        effect="interaction_difference_in_differences",
    )
    close(
        "P1 reduction-style interaction magnitude",
        abs(float(interaction["mean_effect"])),
        0.0030323525276011133,
    )
    close("P1 interaction CI magnitude low", abs(float(interaction["ci95_high"])), 0.002210101756650566)
    close("P1 interaction CI magnitude high", abs(float(interaction["ci95_low"])), 0.0038546032985516607)

    calibration = rows(
        "results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/"
        "calibration_paired_statistics.csv"
    )
    p2_high = one(calibration, metric="high_mae")
    close(
        "P2 high-damage MAE relative improvement (%)",
        float(p2_high["mean_relative_improvement_percent"]),
        7.629745964842028,
    )
    equal("P2 high-damage MAE beneficial splits", int(p2_high["calibrated_wins"]), 10)
    p2_under = one(calibration, metric="high_underestimation_ratio")
    close(
        "P2 high-damage underestimation reduction",
        -float(p2_under["mean_difference"]),
        0.10370590069124352,
    )

    noise = rows(
        "results/sss_fast_revision/matched_noise_robustness_clean_trained/"
        "noise_level_summary.csv"
    )
    p3_expected = {
        "0": 0.022047140089029485,
        "5": 0.04435935226467652,
        "10": 0.11180562156655098,
        "20": 0.28389311833100594,
    }
    for percent, expected in p3_expected.items():
        row = one(noise, method="standard_svr", noise_percent=percent)
        close(f"P3 {percent}% overall MAE", float(row["mae_mean"]), expected)

    bias_rows = rows(
        "results/sss_fast_revision/matched_noise_failure_mechanism/"
        "high_damage_bias_reversal.csv"
    )
    biases: dict[str, float] = {}
    for percent, expected in {
        "0": -0.02133730479685522,
        "5": -0.004789494185237728,
        "10": 0.042574920761648025,
        "20": 0.1285342351836434,
    }.items():
        row = one(bias_rows, method="standard_svr", noise_percent=percent)
        biases[percent] = float(row["high_signed_bias"])
        close(f"P3 {percent}% high-damage signed bias", biases[percent], expected)
    equal("P3 signed-bias reversal between 5% and 10%", biases["5"] < 0 < biases["10"], True)

    prediction = rows(
        "results/sss_fast_revision/matched_noise_failure_mechanism/"
        "prediction_distribution_summary.csv"
    )
    p3_all20 = one(
        prediction,
        method="standard_svr",
        noise_percent="20",
        severity="all",
    )
    p3_high20 = one(
        prediction,
        method="standard_svr",
        noise_percent="20",
        severity="high",
    )
    close("P3 20% overall imposed-bound clipping", float(p3_all20["clip_high_ratio_mean"]), 0.32255555555555554)
    close("P3 20% high-damage imposed-bound clipping", float(p3_high20["clip_high_ratio_mean"]), 0.6408759124087591)
    close("P3 20% high-damage median prediction", float(p3_high20["prediction_q50_mean"]), 0.5)

    layouts = rows(
        "results/sss_fast_revision/exhaustive_sensor_layout_svr/"
        "sensor_layout_results.csv"
    )
    equal("P4 15 layouts detected", len(layouts), 15)
    full = one(layouts, layout_tag="1234")
    close("P4 full-layout MAE", float(full["test_mae"]), 0.022047140089029485)
    equal("P4 full-layout descriptor count", int(full["feature_count"]), 78)
    two_sensor = [row for row in layouts if int(row["sensor_count"]) == 2]
    best_overall = min(two_sensor, key=lambda row: float(row["test_mae"]))["layout_tag"]
    best_high = min(two_sensor, key=lambda row: float(row["test_high_mae"]))["layout_tag"]
    equal("P4 best overall two-sensor layout", best_overall, "12")
    equal("P4 best high-damage two-sensor layout", best_high, "13")
    equal("P4 layout 13 dimension", int(one(layouts, layout_tag="13")["feature_count"]), 31)
    equal("P4 layout 14 dimension", int(one(layouts, layout_tag="14")["feature_count"]), 31)

    edges = rows(
        "results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/"
        "marginal_sensor_edge_bootstrap_ci.csv"
    )
    equal("P5 28 admissible edges detected", len(edges), 28)
    overall_beneficial = sum(as_bool(row["mae_ci_entirely_positive"]) for row in edges)
    high_beneficial = sum(as_bool(row["high_mae_ci_entirely_positive"]) for row in edges)
    equal("P5 all 28 overall effects beneficial", overall_beneficial, 28)
    equal("P5 all 28 high-damage effects beneficial", high_beneficial, 28)

    print()
    if FAILURES:
        print("FROZEN MANUSCRIPT ANCHOR VERIFICATION FAILED")
        for failure in FAILURES:
            print(f"- {failure}")
        print("Stop and document the discrepancy. Do not retune or alter frozen evidence.")
        return 1

    print("ALL FROZEN MANUSCRIPT ANCHORS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

