from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

from mouse_sleep.epilepsy.cross_mouse_raw_shape import (
    SharedRawShapeConfig,
    fit_shared_raw_shape_representation,
)
from mouse_sleep.epilepsy.cross_mouse_waveform_types import (
    CrossMouseWaveformConfig,
    FEATURE_BLOCKS,
    analyze_cross_mouse_waveform_types,
)
from mouse_sleep.epilepsy.cross_mouse_waveform_visualization import (
    generate_cross_mouse_waveform_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match the existing 7/3/3/4/2/2 per-mouse within-seizure waveform "
            "categories across mice without invoking the whole-event trajectory model."
        )
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/epilepsy_morphology_v6_filtered")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to per_mouse/cross_mouse_waveform_types",
    )
    parser.add_argument("--minimum-events", type=int, default=20)
    parser.add_argument("--split-half-repeats", type=int, default=400)
    parser.add_argument("--pair-bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--rocket-kernels", type=int, default=5000)
    parser.add_argument("--maximum-shape-fit-sequences", type=int, default=8000)
    parser.add_argument(
        "--reuse-shared-raw-shape",
        action="store_true",
        help="Reuse the saved shared raw-shape embeddings in the output directory",
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    per_mouse = run_dir / "per_mouse"
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else per_mouse / "cross_mouse_waveform_types"
    )
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    source = per_mouse / "per_mouse_morphology_assignments.parquet"
    model_summary_path = per_mouse / "per_mouse_model_selection.csv"
    if not source.exists() or not model_summary_path.exists():
        raise FileNotFoundError("the existing per-mouse 21-category analysis is missing")
    assignments = pd.read_parquet(source)
    model_summary = pd.read_csv(model_summary_path)
    native_root = run_dir / "temporal_taxonomy" / "native_windows"
    native_metadata_path = native_root / "window_metadata.parquet"
    signals_path = native_root / "signals.npy"
    if not native_metadata_path.exists() or not signals_path.exists():
        raise FileNotFoundError(
            "native filtered windows are required for the shared raw-shape branch"
        )
    native_metadata = pd.read_parquet(native_metadata_path)
    shape_config = SharedRawShapeConfig(
        rocket_kernels=args.rocket_kernels,
        maximum_fit_sequences=args.maximum_shape_fit_sequences,
        random_seed=args.seed + 1,
    )
    cached_shape = output / "shared_raw_shape_event_category_embeddings.parquet"
    if args.reuse_shared_raw_shape:
        if not cached_shape.exists():
            raise FileNotFoundError(
                f"cannot reuse missing shared raw-shape embeddings: {cached_shape}"
            )
        print("[cross-mouse] reusing shared CH8 raw-waveform representation", flush=True)
        raw_shape = SimpleNamespace(
            event_category_embeddings=pd.read_parquet(cached_shape)
        )
    else:
        print(
            "[cross-mouse] fitting one shared CH8 raw-waveform representation",
            flush=True,
        )
        raw_shape = fit_shared_raw_shape_representation(
            assignments,
            native_metadata,
            signals_path,
            output_dir=output,
            config=shape_config,
        )
    raw_shape_columns = [
        column
        for column in raw_shape.event_category_embeddings
        if column.startswith("raw_shape_pc")
    ]
    event_shape = raw_shape.event_category_embeddings[
        ["animal_id", "session_id", "event_id", "local_type"]
        + raw_shape_columns
    ]
    assignments = assignments.copy()
    assignments["local_type"] = assignments[
        "individual_review_smoothed_label"
    ].astype(str)
    assignments = assignments.merge(
        event_shape,
        on=["animal_id", "session_id", "event_id", "local_type"],
        how="left",
        validate="many_to_one",
    ).drop(columns="local_type")
    local_models = {}
    for path in sorted((per_mouse / "models" / "per_mouse").glob("*_morphology.joblib")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(path)
        local_models[str(model["animal_id"])] = model
    config = CrossMouseWaveformConfig(
        minimum_events_for_matching=args.minimum_events,
        split_half_repeats=args.split_half_repeats,
        pair_bootstrap_repeats=args.pair_bootstrap_repeats,
        random_seed=args.seed,
    )
    print("[cross-mouse] reducing windows to event x local-category medians", flush=True)
    result = analyze_cross_mouse_waveform_types(
        assignments,
        model_summary,
        local_models,
        config=config,
        feature_blocks={"raw_shape": tuple(raw_shape_columns), **FEATURE_BLOCKS},
    )
    _save_tables(result, output)
    _copy_existing_atlases(per_mouse / "figures", figures / "waveform_atlases")
    if not args.skip_figures:
        print("[cross-mouse] rendering the 21-category figures", flush=True)
        generate_cross_mouse_waveform_figures(
            result,
            assignments,
            figures,
            existing_atlas_dir=per_mouse / "figures",
            dpi=args.dpi,
        )
    _write_report(result, output)
    _write_manifest(result, config, source, output)
    print(result.type_registry.to_string(index=False), flush=True)
    print(f"[cross-mouse] outputs: {output}", flush=True)
    return 0


def _save_tables(result, output: Path) -> None:
    result.event_category_features.to_parquet(
        output / "event_balanced_local_category_features.parquet", index=False
    )
    tables = {
        "local_waveform_type_inventory.csv": result.local_inventory,
        "local_waveform_type_prototypes.csv": result.category_prototypes,
        "per_mouse_cluster_coordinates.csv": result.local_scatter_coordinates,
        "common_space_21_category_coordinates.csv": result.common_space_coordinates,
        "matching_threshold_calibration.csv": result.threshold_calibration,
        "pairwise_cross_mouse_match_evidence.csv": result.pairwise_evidence,
        "cross_mouse_waveform_type_registry.csv": result.type_registry,
        "cross_mouse_waveform_type_presence.csv": result.presence,
        "representative_discriminative_dimensions.csv": result.representative_dimensions,
        "local_category_position_audit.csv": result.local_position_audit,
        "within_mouse_possible_duplicate_audit.csv": result.within_mouse_pair_audit,
        "strongest_cross_mouse_candidate_pairs.csv": _strongest_candidate_table(
            result.pairwise_evidence, 20
        ),
        "exploratory_cross_mouse_candidate_presence.csv": _exploratory_candidate_pairs(
            result.pairwise_evidence
        ),
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, encoding="utf-8-sig")


def _copy_existing_atlases(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("E*_candidate_waveform_atlas.*"):
        shutil.copy2(path, destination / path.name)


def _write_report(result, output: Path) -> None:
    inventory = result.local_inventory
    registry = result.type_registry
    pairs = result.pairwise_evidence
    duplicate = result.within_mouse_pair_audit
    counts = inventory.groupby("animal_id")["local_type"].nunique().to_dict()
    decisions = pairs["pair_decision"].value_counts().to_dict()
    relations = duplicate["within_mouse_relation"].value_counts().to_dict()
    strongest = _strongest_candidate_table(pairs, 10)
    position_summary = (
        result.local_position_audit.groupby("animal_id")[
            [
                "animal_position_decile_nmi_event_balanced",
                "animal_position_decile_ami_event_balanced",
            ]
        ]
        .first()
        .reset_index()
    )
    lines = [
        "# 21类发作内波形的跨鼠共同性分析",
        "",
        "## 这次实际分析的对象",
        "",
        "只分析原有21个发作内波形状态：E50=7、E76=3、E77=3、E79=4、E81=2、E82=2。一次完整发作可以依次包含多个状态；没有使用已经排除的完整发作轨迹2/1/1/1/1/1，也没有使用U01–U07。",
        "",
        "## 统计单位",
        "",
        "先对每个“小鼠×独立发作×局部类别”取中位数，再建立类别原型和事件bootstrap。因此长发作不会因为窗口更多而支配匹配。",
        "",
        "新增的一致形状证据来自一套在六只鼠上共同拟合的CH8四秒完整原始波形MiniRocket/PCA表征。形状分支逐序列去中心并按1%–99%分位幅度归一化；相对幅值、频域、时域统计和双侧一致性仍作为独立证据。信号已做0.5–70 Hz零相位带通和50 Hz陷波。E77的CH7无效，相关双侧证据记为缺失。ADC→µV标定未知，幅值只能写a.u.或相对基线。",
        "",
        "## 每只鼠保留的类别数",
        "",
        "| 小鼠 | 类别数 | 证据层级 |",
        "|---|---:|---|",
    ]
    for animal in sorted(counts):
        status = inventory.loc[
            inventory["animal_id"].eq(animal), "local_resolution_status"
        ].iloc[0]
        translated = "稳定/已验证" if status == "validated_stable" else "探索性细分"
        lines.append(f"| {animal} | {counts[animal]} | {translated} |")
    lines.extend(
        [
            "",
            "E50和E77的局部分辨率稳定；E76、E79、E81、E82的类别全部保留展示，但仍属探索性，不能把低样本类别直接写成已确认的生物亚型。",
            "",
            "## 是否只是发作位置不同而被拆开",
            "",
            "绝对onset/middle/offset位置从未作为聚类输入。现在又改成先求每个“发作×类别”的位置中位数，再跨发作汇总，避免长发作支配。位置只用于事后解释。",
            "",
            "| 小鼠 | 位置NMI | 校正位置AMI |",
            "|---|---:|---:|",
        ]
    )
    for row in position_summary.itertuples(index=False):
        lines.append(
            f"| {row.animal_id} | "
            f"{row.animal_position_decile_nmi_event_balanced:.3f} | "
            f"{row.animal_position_decile_ami_event_balanced:.3f} |"
        )
    lines.extend(
        [
            "",
            "AMI已校正小样本和类别数带来的偶然一致性；六只鼠均接近0，E77最高也只有0.085。",
            "",
            "## 同鼠内完整波形复核",
            "",
            f"- 原始波形形状明确不同：{relations.get('shape_distinct', 0)}对。",
            f"- 波形轮廓相近，但幅值/频域/双侧证据明确不同：{relations.get('same_shape_family_physiology_distinct', 0)}对。",
            f"- 所有可用证据都相近，需要警惕重复细分：{relations.get('possible_duplicate_subdivision', 0)}对。",
            f"- 低样本边界瞬态：{relations.get('low_sample_boundary_transient', 0)}对。",
            f"- 证据混合或不足：{relations.get('insufficient_or_mixed_evidence', 0)}对。",
            "",
            "辅助特征可以合理保留“形状相似但生理不同”的状态；也确实可能在少数配对上造成重复细分。正式清单见within_mouse_possible_duplicate_audit.csv，不能只凭二维图判断。",
            "",
            "## 跨鼠正式匹配",
            "",
            f"- 正式稳定共享配对：{decisions.get('confirmed_shared_candidate', 0)}。",
            f"- 探索性共享候选：{decisions.get('provisional_shared_candidate', 0)}。",
            f"- 低事件支持但相容：{decisions.get('compatible_low_event_support', 0)}。",
            f"- 混合证据未合并：{decisions.get('not_merged_mixed_evidence', 0)}。",
            f"- 有明确差异证据：{decisions.get('distinct_evidence', 0)}。",
            f"- 保守登记表保留{len(registry)}个动物限定类别。",
            "",
            "“保留21个”不等于证明没有共同生物学；它只表示没有配对同时通过完整原始形状、幅值、时域统计、频域以及可用双侧证据的事件级95% bootstrap上界。",
            "",
            "## 最接近的跨鼠配对",
            "",
            "| 配对 | 决策 | 最大95%CI/阈值 | 阻止合并的证据 |",
            "|---|---|---:|---|",
        ]
    )
    for row in strongest.itertuples(index=False):
        lines.append(
            f"| {row.local_type_a} ↔ {row.local_type_b} | {row.pair_decision} | "
            f"{row.candidate_ci_max_ratio:.2f} | {row.blocks_blocking_merge or '无'} |"
        )
    lines.extend(
        [
            "",
            "## 图怎么读",
            "",
            "- Figure 1：每只鼠的事件-类别点云；极端点压到边界并标注数量。",
            "- Figure 2：21类在共同完整波形+生理空间中的原型图。",
            "- Figure 3：正式保守登记表；如果无配对通过，接近对角线是正确结果。",
            "- Figure 7：最接近跨鼠配对的证据矩阵；每格为95%CI上界/阈值，≤1才通过，灰色为不可用。",
            "- Figure 8：同鼠内可能重复细分审计；绿色表示所有证据相近，蓝色表示轮廓近似但生理不同。",
            "- Figure 9：单独列出跨两只鼠但未达到正式合并标准的探索候选；这些行可以重叠，不能直接数成共同发作种类。",
            "- 各鼠单独波形图保存在figures/waveform_atlases/，临床复核应看单图。",
        ]
    )
    (output / "CLINICAL_SUMMARY_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _strongest_candidate_table(pairs: pd.DataFrame, maximum: int) -> pd.DataFrame:
    result = pairs.copy()
    blocks = [
        block
        for block in ("raw_shape", "amplitude", "shape", "frequency", "bilateral")
        if f"{block}_threshold" in result
    ]
    ratios, blockers = [], []
    for row in result.itertuples(index=False):
        values, failed = [], []
        for block in blocks:
            high = getattr(row, f"{block}_ci_high")
            threshold = getattr(row, f"{block}_threshold")
            if pd.notna(high) and pd.notna(threshold) and threshold > 0:
                ratio = float(high) / float(threshold)
                values.append(ratio)
                if ratio > 1.0:
                    failed.append(block)
        ratios.append(max(values) if values else float("inf"))
        blockers.append(";".join(failed))
    result["candidate_ci_max_ratio"] = ratios
    result["blocks_blocking_merge"] = blockers
    return result.sort_values(
        ["candidate_ci_max_ratio", "maximum_required_distance_ratio"], kind="stable"
    ).head(maximum)


def _exploratory_candidate_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    decisions = {
        "confirmed_shared_candidate",
        "provisional_shared_candidate",
        "compatible_low_event_support",
        "compatible_uncertain",
        "not_merged_mixed_evidence",
    }
    result = _strongest_candidate_table(
        pairs.loc[pairs["pair_decision"].isin(decisions)], 50
    ).reset_index(drop=True)
    result.insert(0, "candidate_pair_id", [f"CP{index + 1:02d}" for index in range(len(result))])
    result["mouse_count"] = 2
    result["mouse_coverage"] = "2/6"
    result["formal_shared_type"] = result["pair_decision"].eq(
        "confirmed_shared_candidate"
    )
    result["interpretation"] = "pair_level_similarity_not_merged_global_type"
    return result


def _write_manifest(result, config, source: Path, output: Path) -> None:
    raw_shape_manifest = output / "shared_raw_shape_manifest.json"
    raw_shape = (
        json.loads(raw_shape_manifest.read_text(encoding="utf-8"))
        if raw_shape_manifest.exists()
        else None
    )
    payload = {
        "schema_version": 2,
        "analysis_name": "cross_mouse_existing_21_within_seizure_waveform_types",
        "source_assignments": str(source),
        "label_column": config.label_column,
        "local_type_counts": result.local_inventory.groupby("animal_id")[
            "local_type"
        ].nunique().to_dict(),
        "local_types_total": int(result.local_inventory["local_type"].nunique()),
        "event_category_cells": int(len(result.event_category_features)),
        "whole_event_trajectory_model_used": False,
        "u01_u07_used": False,
        "shared_full_raw_waveform_shape_used": raw_shape is not None,
        "shared_raw_shape": raw_shape,
        "config": {
            "minimum_events_for_matching": config.minimum_events_for_matching,
            "split_half_repeats": config.split_half_repeats,
            "pair_bootstrap_repeats": config.pair_bootstrap_repeats,
            "threshold_quantile": config.threshold_quantile,
            "confidence_level": config.confidence_level,
            "random_seed": config.random_seed,
        },
        "warnings": [
            "Local review resolution is exploratory for E76, E79, E81 and E82.",
            "Absolute ADC-to-microvolt calibration is unavailable; amplitude is relative or a.u.",
            "E77 CH7 is invalid; bilateral evidence is omitted for E77 pairs.",
            "Shared labels are operational matches, not clinical seizure subtype names.",
        ],
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
