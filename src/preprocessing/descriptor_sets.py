"""
Canonical descriptor-set definitions for the SSS fast revision.

This module is the single source of truth for descriptor provenance
and feature-set membership. Other experiment scripts should import
these definitions instead of implementing independent string rules.

统一管理 SSS 快速修订版的描述符来源和特征集合。
后续实验脚本不得再自行重复编写特征筛选规则。
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd


class DescriptorSetName(str, Enum):
    """Supported descriptor-set names. / 支持的描述符集合名称。"""

    ORACLE_FULL = "oracle_full"
    LEGACY_NO_META = "legacy_no_meta"
    SIGNAL_DERIVED_WITH_GROUND = "signal_derived_with_ground"
    STRUCTURAL_RESPONSE_ONLY = "structural_response_only"


class DescriptorSource(str, Enum):
    """Descriptor information source. / 描述符信息来源。"""

    GENERATOR_METADATA = "generator_metadata"
    GENERATOR_FREQUENCY_DERIVED = "generator_frequency_derived"
    MEASURED_GROUND_SIGNAL = "measured_ground_signal"
    STRUCTURAL_RESPONSE_SIGNAL = "structural_response_signal"


# Six descriptors explicitly copied from simulation conditions
# or constructed from simulation inputs and a fixed healthy frequency.
#
# 六个显式使用仿真生成条件或固定健康频率的描述符。
EXPLICIT_GENERATOR_FEATURES = frozenset(
    {
        "input_amplitude_g",
        "input_frequency_hz",
        "input_noise_level",
        "input_frequency_to_healthy_mode1_ratio",
        "input_frequency_distance_to_healthy_mode1",
        "input_mode1_resonance_indicator",
    }
)


def normalize_feature_name(name: str) -> str:
    """Normalize one descriptor name. / 规范化描述符名称。"""
    return str(name).strip().lower()


def is_explicit_generator_feature(name: str) -> bool:
    """
    Return whether the descriptor explicitly uses generator metadata.

    判断特征是否显式使用仿真生成器元数据。
    """
    return normalize_feature_name(name) in EXPLICIT_GENERATOR_FEATURES


def is_generator_frequency_derived(name: str) -> bool:
    """
    Identify response descriptors divided by the generator frequency.

    These eight descriptors use the exact simulation parameter
    frequency_hz rather than a frequency estimated from the measured
    ground-motion signal.

    识别直接使用仿真生成器 frequency_hz 作为分母的八个派生特征。
    """
    normalized = normalize_feature_name(name)

    return normalized.endswith(
        "_dominant_frequency_to_input_ratio"
    ) or normalized.endswith(
        "_centroid_to_input_ratio"
    )


def is_ground_signal_derived(name: str) -> bool:
    """
    Identify descriptors requiring the measured ground/input signal.

    Included:
    - ground statistics and spectral descriptors;
    - floor-to-ground amplification descriptors;
    - first-storey response relative to ground;
    - floor-to-ground correlations.

    判断是否必须依赖地面输入传感器时程。
    """
    normalized = normalize_feature_name(name)

    if normalized.startswith("ground_"):
        return True

    if "_ground_" in normalized:
        return True

    if normalized.startswith("story_1_relative_to_lower_"):
        return True

    return False


def descriptor_source(name: str) -> DescriptorSource:
    """
    Resolve the information source of one descriptor.

    确定单个描述符的原始信息来源。
    """
    if is_explicit_generator_feature(name):
        return DescriptorSource.GENERATOR_METADATA

    if is_generator_frequency_derived(name):
        return DescriptorSource.GENERATOR_FREQUENCY_DERIVED

    if is_ground_signal_derived(name):
        return DescriptorSource.MEASURED_GROUND_SIGNAL

    return DescriptorSource.STRUCTURAL_RESPONSE_SIGNAL


def resolve_descriptor_set_name(
    descriptor_set: str | DescriptorSetName,
) -> DescriptorSetName:
    """Resolve a string or enum to DescriptorSetName."""
    if isinstance(descriptor_set, DescriptorSetName):
        return descriptor_set

    normalized = str(descriptor_set).strip().lower()

    aliases = {
        "full": DescriptorSetName.ORACLE_FULL,
        "oracle": DescriptorSetName.ORACLE_FULL,
        "oracle_full": DescriptorSetName.ORACLE_FULL,
        "no_meta": DescriptorSetName.LEGACY_NO_META,
        "physics_no_meta_core": DescriptorSetName.LEGACY_NO_META,
        "legacy_no_meta": DescriptorSetName.LEGACY_NO_META,
        "signal_derived": (
            DescriptorSetName.SIGNAL_DERIVED_WITH_GROUND
        ),
        "signal_derived_with_ground": (
            DescriptorSetName.SIGNAL_DERIVED_WITH_GROUND
        ),
        "response_only": (
            DescriptorSetName.STRUCTURAL_RESPONSE_ONLY
        ),
        "structural_response_only": (
            DescriptorSetName.STRUCTURAL_RESPONSE_ONLY
        ),
    }

    if normalized not in aliases:
        supported = ", ".join(
            descriptor.value
            for descriptor in DescriptorSetName
        )
        raise ValueError(
            f"Unknown descriptor set: {descriptor_set}. "
            f"Supported canonical sets: {supported}"
        )

    return aliases[normalized]


def descriptor_is_selected(
    name: str,
    descriptor_set: str | DescriptorSetName,
) -> bool:
    """
    Return whether one descriptor belongs to a named set.

    判断单个描述符是否属于指定集合。
    """
    resolved = resolve_descriptor_set_name(descriptor_set)
    source = descriptor_source(name)

    if resolved is DescriptorSetName.ORACLE_FULL:
        return True

    if resolved is DescriptorSetName.LEGACY_NO_META:
        # Preserve the old AES rule exactly:
        # remove only the six names beginning with "input_".
        return not normalize_feature_name(name).startswith("input_")

    if resolved is DescriptorSetName.SIGNAL_DERIVED_WITH_GROUND:
        return source in {
            DescriptorSource.MEASURED_GROUND_SIGNAL,
            DescriptorSource.STRUCTURAL_RESPONSE_SIGNAL,
        }

    if resolved is DescriptorSetName.STRUCTURAL_RESPONSE_ONLY:
        return source is DescriptorSource.STRUCTURAL_RESPONSE_SIGNAL

    raise RuntimeError(f"Unhandled descriptor set: {resolved}")


def get_descriptor_indices(
    feature_names: Sequence[str],
    descriptor_set: str | DescriptorSetName,
) -> np.ndarray:
    """
    Return selected descriptor column indices.

    返回指定描述符集合的列索引。
    """
    normalized_names = [
        str(name).strip()
        for name in feature_names
    ]

    if len(normalized_names) != len(set(normalized_names)):
        duplicates = pd.Series(normalized_names)
        duplicates = duplicates[
            duplicates.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            "Duplicate descriptor names detected: "
            f"{sorted(set(duplicates))}"
        )

    mask = np.asarray(
        [
            descriptor_is_selected(name, descriptor_set)
            for name in normalized_names
        ],
        dtype=bool,
    )

    indices = np.flatnonzero(mask)

    if indices.size == 0:
        raise ValueError(
            f"Descriptor set '{descriptor_set}' selected zero columns."
        )

    return indices


def build_descriptor_manifest(
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    Build a complete provenance and set-membership table.

    生成完整描述符来源及集合归属清单。
    """
    names = [
        str(name).strip()
        for name in feature_names
    ]

    rows: list[dict[str, object]] = []

    for index, name in enumerate(names):
        source = descriptor_source(name)

        row: dict[str, object] = {
            "feature_index": index,
            "feature_name": name,
            "descriptor_source": source.value,
        }

        for descriptor_set in DescriptorSetName:
            row[descriptor_set.value] = descriptor_is_selected(
                name,
                descriptor_set,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def descriptor_set_counts(
    feature_names: Sequence[str],
) -> dict[str, int]:
    """Return descriptor count for every canonical set."""
    return {
        descriptor_set.value: int(
            get_descriptor_indices(
                feature_names,
                descriptor_set,
            ).size
        )
        for descriptor_set in DescriptorSetName
    }


def descriptor_source_counts(
    feature_names: Sequence[str],
) -> dict[str, int]:
    """Return descriptor count for every information source."""
    counts = {
        source.value: 0
        for source in DescriptorSource
    }

    for name in feature_names:
        source = descriptor_source(name)
        counts[source.value] += 1

    return counts
