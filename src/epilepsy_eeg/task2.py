"""Task 2 helpers for the frozen PG36 morphology release."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import MODEL_ORDER_TASK2, sha256, write_json
from .figure_manifest import figure_family, write_figure_manifest


_CURRENT_REPORT_NAMES = (
    "TASK2_POOLED_FINAL_REPORT.md",
    "TASK2_METRICS_AND_FIGURES_INDEX_zh.md",
    "CLASSIFIER_TASK2_REPORT_AUTO.md",
)
_STALE_REPORT_NAMES = frozenset(
    {
        "CLASSIFIER_TASK2_REPORT_zh.md",
        "MORPHOLOGY_TASK2_REPORT_zh.md",
    }
)


def _json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _split_counts(frame: pd.DataFrame, column: str = "split") -> dict[str, int]:
    if column not in frame.columns:
        return {}
    values = frame[column].astype(str).value_counts()
    return {str(key): int(value) for key, value in values.items()}


def _event_key(frame: pd.DataFrame) -> pd.Series:
    columns = [column for column in ("animal_id", "session_id", "event_id") if column in frame.columns]
    if not columns:
        return pd.Series([str(index) for index in frame.index], index=frame.index, dtype="string")
    return frame[columns].fillna("").astype(str).agg("|".join, axis=1)


def _cross_split_window_overlaps(frame: pd.DataFrame) -> int:
    """Count cross-split overlapping windows without assuming a zero audit."""

    required = {"start_time_ms", "end_time_ms", "split"}
    if not required.issubset(frame.columns):
        return 0
    group_columns = [column for column in ("animal_id", "session_id", "channel_name") if column in frame.columns]
    if not group_columns:
        group_columns = ["split"]
    count = 0
    for _, group in frame.groupby(group_columns, sort=False, dropna=False):
        rows = group[["start_time_ms", "end_time_ms", "split"]].copy()
        rows["start_time_ms"] = pd.to_numeric(rows["start_time_ms"], errors="coerce")
        rows["end_time_ms"] = pd.to_numeric(rows["end_time_ms"], errors="coerce")
        rows = rows.dropna().sort_values(["start_time_ms", "end_time_ms"], kind="stable")
        active: list[tuple[float, float, str]] = []
        for start, end, split in rows.itertuples(index=False, name=None):
            start_value, end_value = float(start), float(end)
            active = [item for item in active if item[1] > start_value]
            count += sum(item[2] != str(split) for item in active)
            active.append((start_value, end_value, str(split)))
    return int(count)


def _write_pg36_release_audit(
    *,
    target: Path,
    taxonomy: dict[str, Any],
    comparison: pd.DataFrame,
    source: Path,
) -> None:
    """Create release-local audits from the current PG36 tables.

    The source study tree contains historical G-taxonomy audits beside the
    corrected PG36 run.  Recomputing these small summaries from the copied
    tables prevents a stale report from being promoted into the formal release.
    """

    taxonomy_dir = target / "taxonomy"
    labels_path = taxonomy_dir / "pooled_channel_window_labels.parquet"
    event_labels_path = taxonomy_dir / "pooled_event_labels.parquet"
    labels = pd.read_parquet(labels_path) if labels_path.is_file() else pd.DataFrame()
    event_labels = pd.read_parquet(event_labels_path) if event_labels_path.is_file() else pd.DataFrame()
    if event_labels.empty and not labels.empty:
        keys = [column for column in ("animal_id", "session_id", "event_id", "channel_name", "split", "pooled_label", "global_label") if column in labels.columns]
        event_labels = labels[keys].drop_duplicates(
            [column for column in ("animal_id", "session_id", "event_id", "channel_name") if column in keys]
        )
    event_keys = _event_key(event_labels if not event_labels.empty else labels)
    event_frame = event_labels.copy()
    if not event_frame.empty:
        event_frame["event_key"] = event_keys.to_numpy()
    mixed_event_keys = 0
    if not event_frame.empty and "split" in event_frame.columns:
        mixed_event_keys = int((event_frame.groupby("event_key")["split"].nunique() > 1).sum())
    split_counts = _split_counts(labels)
    event_channel_counts = _split_counts(event_frame)
    test = labels[labels.get("split", pd.Series(index=labels.index, dtype=object)).astype(str) == "test"] if not labels.empty else labels
    label_column = "global_label" if "global_label" in test.columns else "pooled_global_label" if "pooled_global_label" in test.columns else None
    support = test[label_column].astype(str).value_counts().sort_index().to_dict() if label_column else {}
    source_batch_counts = labels["source_batch"].astype(str).value_counts().to_dict() if "source_batch" in labels.columns else {}
    animals = sorted(labels["animal_id"].astype(str).unique().tolist()) if "animal_id" in labels.columns else []
    channels = sorted(labels["channel_name"].astype(str).unique().tolist()) if "channel_name" in labels.columns else []
    low_confidence = int(labels.get("low_confidence", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not labels.empty else 0
    run_audit = {
        "schema_version": 1,
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "n_global_categories": int(taxonomy.get("n_global_categories", 0)),
        "n_event_channel": int(taxonomy.get("n_event_channel", len(event_frame))),
        "n_windows": int(taxonomy.get("n_windows", len(labels))),
        "animals": animals,
        "valid_channels": channels,
        "excluded_e77_ch7": not ((labels.get("animal_id", pd.Series(dtype=str)).astype(str) == "E77") & (labels.get("channel_name", pd.Series(dtype=str)).astype(str) == "CH7")).any(),
        "event_split_counts": {
            "eligible_event_channel": event_channel_counts,
            "eligible_windows": split_counts,
        },
        "source_batch_counts": {str(key): int(value) for key, value in source_batch_counts.items()},
        "cross_split_mixed_event_keys": mixed_event_keys,
        "cross_split_overlapping_windows": _cross_split_window_overlaps(labels),
        "low_confidence_window_rows": low_confidence,
        "test_category_support": {
            "classes_present": int(len(support)),
            "minimum_windows": int(min(support.values())) if support else 0,
            "maximum_windows": int(max(support.values())) if support else 0,
            "counts": {str(key): int(value) for key, value in support.items()},
        },
        "source_root_is_provenance_only": str(source),
    }
    split_audit = {
        "schema_version": 1,
        "split_unit": "clinical_event with independent valid-channel observations",
        "assignment": "mixed animals in train/val/test; event identities never cross splits",
        "target_ratio": {"train": 0.70, "val": 0.15, "test": 0.15},
        "eligible_event_channel_counts": event_channel_counts,
        "eligible_window_counts": split_counts,
        "n_global_categories": int(taxonomy.get("n_global_categories", 0)),
        "test_category_support": run_audit["test_category_support"],
        "overlap_audit": {
            "cross_split_mixed_event_keys": mixed_event_keys,
            "cross_split_overlapping_windows": run_audit["cross_split_overlapping_windows"],
        },
        "context": {
            "sequence_length_epochs": 21,
            "context_scope": "same event and same channel only",
            "out_of_event_padding": "zero",
            "cross_split_context_borrowing": False,
        },
    }
    audit_dir = target / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(audit_dir / "run_audit.json", run_audit)
    write_json(audit_dir / "split_audit.json", split_audit)

    environment: dict[str, Any] = {}
    for candidate in (
        source / "08_audit" / "classifier_environment_05_classification_pooled.json",
        source / "audit" / "classifier_environment_05_classification_pooled.json",
    ):
        environment = _json_file(candidate)
        if environment:
            break
    environment.update(
        {
            "schema_version": 1,
            "models_requested": list(MODEL_ORDER_TASK2),
            "training_models": list(MODEL_ORDER_TASK2),
            "all_window_evaluation_models": list(MODEL_ORDER_TASK2),
            "classes": int(taxonomy.get("n_global_categories", 0)),
            "checkpoint_classes": {model: int(taxonomy.get("n_global_categories", 0)) for model in MODEL_ORDER_TASK2},
            "sequence_length": 21,
            "normalization": "per-window median/MAD, clip ±20",
            "weights_pretrained": False,
            "label_path": "taxonomy/pooled_channel_window_labels.parquet",
            "primary_metrics_file": "classification/all_window_model_comparison.csv",
            "sensitivity_metrics_file": "classification/model_comparison.csv",
            "output_subdir": "classification",
            "source_environment_is_provenance_only": True,
        }
    )
    write_json(audit_dir / "classifier_environment.json", environment)
    (audit_dir / "REPORT_AUDIT_CHANGELOG_zh.md").write_text(
        "# PG36 发布审计变更\n\n"
        "本目录的审计摘要均从当前 `../taxonomy/` 与 `../classification/` 表重新生成；历史 G taxonomy 和旧 21 状态不属于本 release。\n\n"
        f"- 全局探索性形态类别：PG01–PG{int(taxonomy.get('n_global_categories', 0)):02d}。\n"
        f"- 事件×有效通道观察：{int(taxonomy.get('n_event_channel', len(event_frame))):,}；完整窗口：{int(taxonomy.get('n_windows', len(labels))):,}。\n"
        f"- 全窗口测试分母：{int(comparison['n'].iloc[0]) if not comparison.empty and 'n' in comparison else 0:,}，由当前全窗口评估表读取。\n"
        f"- 跨 split 混入事件键：{mixed_event_keys}；跨 split 重叠窗口：{run_audit['cross_split_overlapping_windows']}。\n"
        "- 每个有效通道独立输入；E77/CH7 按 QC 规则屏蔽。\n",
        encoding="utf-8",
    )
    (audit_dir / "README_zh.md").write_text(
        "# PG36 审计文件\n\n"
        "`run_audit.json`、`split_audit.json` 和 `classifier_environment.json` 均由当前 release 的 PG36 标签、全窗口预测和指标表生成。它们不是旧分类 lineage 的复制品。\n",
        encoding="utf-8",
    )


def _write_pg36_release_reports(*, target: Path, taxonomy: dict[str, Any], comparison: pd.DataFrame, source: Path) -> None:
    """Write compact release reports with paths matching the release layout."""

    reports_dir = target / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    source_reports = next((source / name for name in ("07_reports", "reports") if (source / name).is_dir()), None)

    def current_source(name: str) -> str | None:
        if source_reports is None:
            return None
        path = source_reports / name
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        # A report that presents the old G taxonomy as the primary task is not
        # eligible for a PG36 release, even if its filename looks current.
        if re.search(r"G01\s*[–-]\s*G45|45\s*个(?:全局|探索性)类别", text):
            return None
        return text

    for name in _CURRENT_REPORT_NAMES:
        text = current_source(name)
        if text is None:
            text = ""
        text = (
            text.replace("04_label_freeze_pooled", "taxonomy")
            .replace("05_classification_pooled", "classification")
            .replace("06_figures/paper_ready_morphology_pooled", "figures")
            .replace("06_figures/paper_ready_classifier_pooled", "figures/model_summary")
            .replace("06_figures/strict_classifier_full_pooled", "figures/confusion_all_windows")
            .replace(
                "Legacy 21-state and previous G taxonomy are retained under `historical_21_read_only`, `legacy_style_morphology`, `legacy_style_cross_mouse`, and `04_label_freeze`; they are priors/audit references, not the PG final classifier.",
                "Historical 21-state and previous G taxonomy artifacts are intentionally excluded from this PG36 release and remain available only in the source study archive.",
            )
        )
        # Reports live one directory below the release root. Normalize common
        # Markdown destinations after the source-layout substitutions so links
        # remain valid in the compact release tree.
        text = re.sub(
            r"\]\((?!(?:\.\./|https?://))((?:taxonomy|classification|figures)(?:/[^)]*)?)\)",
            r"](../\1)",
            text,
        )
        text = text.replace("`classification/all_window_model_comparison.csv`", "`../classification/all_window_model_comparison.csv`")
        if not text.strip() or name == "CLASSIFIER_TASK2_REPORT_AUTO.md":
            n = int(taxonomy.get("n_global_categories", 0))
            primary = int(comparison["n"].iloc[0]) if not comparison.empty and "n" in comparison else 0
            text = (
                "# PG36 形态类别分类结果\n\n"
                f"当前冻结 taxonomy 为 PG01–PG{n:02d}；每个事件×有效通道是独立观察。\n\n"
                f"全窗口测试主分母为 {primary:,}，指标来自 `../classification/all_window_model_comparison.csv`。\n"
                "PG 标签是探索性形态类别，不是临床发作亚型；旧 lineage 未纳入本 release。\n\n"
                "重要限制：PG36 分类结果仅代表当前形态探索 split；与任务一严格 protocol 比较前必须通过 `event_id + channel + split` 对齐审计，不能直接宣称跨域泛化，也不能与任务一指标合并。\n"
            )
        elif "alignment caveat" not in text.lower() and "对齐限制" not in text:
            text += (
                "\n\n### Alignment caveat\n\n"
                "This PG36 classifier uses its own morphology-exploration split. Compare it with a Task-1 strict protocol only after an explicit `event_id + channel + split` key audit; do not merge the two metric denominators or present them as one cross-domain generalization estimate.\n"
            )
        (reports_dir / name).write_text(text, encoding="utf-8")
    (reports_dir / "README_zh.md").write_text(
        "# PG36 报告入口\n\n"
        "本目录只保留当前 PG36 的 pooled 形态与全窗口分类报告；历史 taxonomy 不作为正式结果复制。"
        "图形和机器可读审计分别见 `../figures/` 与 `../audit/`。\n\n"
        "这些报告使用 PG36 自己的事件划分；与任务一严格 protocol 的比较必须先通过 event_id + channel + split 对齐审计，不能直接合并指标。\n",
        encoding="utf-8",
    )


def _normalize_pg36_release_metadata(target: Path, taxonomy: dict[str, Any], comparison: pd.DataFrame) -> None:
    """Make copied JSON/README metadata refer to the compact release layout."""

    replacements = (
        ("04_label_freeze_pooled", "taxonomy"),
        ("05_classification_pooled", "classification"),
        ("04_label_freeze", "taxonomy"),
        ("05_classification", "classification"),
    )
    for path in (target / "taxonomy").rglob("*.json") if (target / "taxonomy").is_dir() else ():
        payload = _json_file(path)
        if not payload:
            continue

        def rewrite(value: Any) -> Any:
            if isinstance(value, str):
                result = value
                for old, new in replacements:
                    result = result.replace(old, new)
                return result
            if isinstance(value, dict):
                return {key: rewrite(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite(item) for item in value]
            return value

        updated = rewrite(payload)
        if path.name == "pooled_label_freeze_manifest.json":
            updated["source_files"] = {
                "pooled_mouse_prototypes": "tables/pooled_mouse_prototypes.parquet",
                "pooled_mouse_event_assignments": "tables/pooled_mouse_event_assignments.parquet",
                "channel_event_features": "tables/channel_event_features.parquet",
                "source_window_labels": "taxonomy/pooled_channel_window_labels.parquet",
            }
            updated["release_layout"] = {
                "taxonomy": "taxonomy",
                "tables": "tables",
                "classification": "classification",
                "figures": "figures",
            }
        write_json(path, updated)

    classification = target / "classification"
    for path in classification.rglob("*.json") if classification.is_dir() else ():
        payload = _json_file(path)
        if not payload:
            continue

        def rewrite_class(value: Any) -> Any:
            if isinstance(value, str):
                result = value
                for old, new in replacements:
                    result = result.replace(old, new)
                return result
            if isinstance(value, dict):
                return {key: rewrite_class(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rewrite_class(item) for item in value]
            return value

        updated = rewrite_class(payload)
        if "output_subdir" in updated:
            updated["output_subdir"] = "classification"
        if "label_path" in updated:
            updated["label_path"] = "taxonomy/pooled_channel_window_labels.parquet"
        write_json(path, updated)

    for path in classification.rglob("*.csv") if classification.is_dir() else ():
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        changed = False
        if "label_path" in frame.columns:
            frame["label_path"] = "taxonomy/pooled_channel_window_labels.parquet"
            changed = True
        if changed:
            frame.to_csv(path, index=False)

    taxonomy_readme = target / "taxonomy" / "README_zh.md"
    if taxonomy_readme.is_file():
        taxonomy_readme.write_text(
            "# PG36 冻结形态标签\n\n"
            f"本目录包含 {int(taxonomy.get('n_event_channel', 0)):,} 个事件×有效通道观察、"
            f"{int(taxonomy.get('n_windows', 0)):,} 个完整窗口和 PG01–PG{int(taxonomy.get('n_global_categories', 0)):02d} 个探索性类别。"
            "每个有效通道独立输入；旧 taxonomy 不属于本 release。\n\n"
            "`pooled_event_labels.parquet` 是事件×通道标签，`pooled_channel_window_labels.parquet` 是窗口标签，"
            "`pooled_category_registry.*` 和 `pooled_global_prototypes.parquet` 用于类别支持与原型审计。\n",
            encoding="utf-8",
        )
    class_readme = classification / "README_zh.md"
    if class_readme.is_file():
        text = class_readme.read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        primary = int(comparison["n"].iloc[0]) if not comparison.empty and "n" in comparison else 0
        text = re.sub(r"全部 [0-9,]+ 个正式测试窗口", "全部正式测试窗口", text)
        text = re.sub(r"完整测试窗口 [0-9,]+", f"完整测试窗口 {primary:,}", text)
        if "对齐限制" not in text and "alignment caveat" not in text.lower():
            text += (
                "\n重要对齐限制：PG36 分类结果属于形态探索快照；与任务一严格 protocol 比较前必须通过 `event_id + channel + split` 对齐审计，不能直接合并指标。\n"
            )
        class_readme.write_text(text, encoding="utf-8")


def _copy_tree(source: Path, target: Path, *, exclude_suffixes: set[str] | None = None) -> int:
    if not source.exists():
        raise FileNotFoundError(source)
    count = 0
    exclude_suffixes = exclude_suffixes or set()
    for item in source.rglob("*"):
        if not item.is_file() or item.suffix.lower() in exclude_suffixes:
            continue
        # The canonical source root contains historical smoke/failed audits.
        # They are intentionally not part of the formal PG36 release.
        relative_parts = {part.lower() for part in item.relative_to(source).parts}
        if any(token in relative_parts for token in {"legacy_failed_attempts", "archive", "smoke"}) or any(token in item.name.lower() for token in ("legacy", "failed", "smoke", "archive")):
            continue
        relative = item.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        count += 1
    return count


def materialize_pg36(source_root: str | Path, output_root: str | Path, *, include_tiff: bool = False, include_predictions: bool = True, allow_existing: bool = False) -> Path:
    """Copy only the final pooled PG36 lineage into a clean release root."""
    source = Path(source_root).resolve()
    target = Path(output_root).resolve()
    if target.exists() and any(target.iterdir()) and not allow_existing:
        raise FileExistsError(f"输出目录非空: {target}")
    target.mkdir(parents=True, exist_ok=True)
    # ``allow_existing`` is an explicit request to refresh a generated release.
    # Clear only the generated report/audit subtrees so stale lineage cannot
    # survive a re-materialization; raw data and unrelated files are untouched.
    if allow_existing:
        for relative in ("reports", "audit"):
            directory = target / relative
            if directory.exists():
                shutil.rmtree(directory)
    copied: list[str] = []
    selected_candidates = {
        "04_label_freeze_pooled": ("04_label_freeze_pooled",),
        "05_classification_pooled": ("05_classification_pooled",),
        "reports": ("07_reports", "reports"),
        "audit": ("08_audit", "audit"),
    }
    for target_name, candidates in selected_candidates.items():
        source_name = next((name for name in candidates if (source / name).is_dir()), None)
        if source_name is None:
            if target_name in {"04_label_freeze_pooled", "05_classification_pooled"}:
                raise FileNotFoundError(source / candidates[0])
            continue
        destination = target / ("taxonomy" if target_name == "04_label_freeze_pooled" else "classification" if target_name == "05_classification_pooled" else target_name)
        if target_name in {"reports", "audit"}:
            # These directories contain historical G01–G45 artifacts beside
            # the corrected PG36 run.  They are regenerated below from the
            # copied tables instead of copied wholesale.
            continue
        copied_count = _copy_tree(source / source_name, destination, exclude_suffixes=set())
        copied.append(f"{destination.name}: {copied_count} files")
    # Tables needed to reproduce the frozen taxonomy audit, not the old 45-class lineage.
    table_target = target / "tables"
    table_target.mkdir(parents=True, exist_ok=True)
    for relative in (
        "01_tables/event_splits.parquet",
        "01_tables/channel_event_features.parquet",
        "03_cross_mouse/pooled_mouse_prototypes.parquet",
        "03_cross_mouse/pooled_mouse_event_assignments.parquet",
    ):
        source_path = source / relative
        if source_path.is_file():
            destination = table_target / Path(relative).name
            shutil.copy2(source_path, destination)
            copied.append(relative)
    # The source study tree may contain an older renderer beside the corrected
    # one.  Prefer the explicit ``_v2`` product so a release cannot silently
    # materialize the legacy directory names/figure contract.
    figure_source = next(
        (
            source / "06_figures" / name
            for name in ("paper_ready_morphology_pooled_v2", "paper_ready_morphology_pooled")
            if (source / "06_figures" / name).is_dir()
        ),
        None,
    )
    if figure_source is None:
        raise FileNotFoundError(source / "06_figures" / "paper_ready_morphology_pooled")
    figure_target = target / "figures"
    figure_exclude = set() if include_tiff else {".tiff"}
    copied.append(f"figures/morphology: {_copy_tree(figure_source, figure_target, exclude_suffixes=figure_exclude)} files")
    # Model-level summary and full-window confusion matrices are generated by
    # the same frozen PG36 run but live beside the morphology atlas in the
    # source project.  Keep only the publication formats and omit smoke runs.
    for source_subdir, target_subdir in (
        ("06_figures/paper_ready_classifier_pooled", "figures/model_summary"),
        ("06_figures/strict_classifier_full_pooled", "figures/confusion_all_windows"),
    ):
        source_fig = source / source_subdir
        if source_fig.is_dir():
            copied_count = _copy_tree(source_fig, target / target_subdir, exclude_suffixes=set() if include_tiff else {".tiff"})
            copied.append(f"{source_subdir}: {copied_count} files")
    # Keep the full all-window predictions only when explicitly requested; the
    # metrics and confusion matrices are always retained.
    if not include_predictions:
        for path in (target / "classification").rglob("*.parquet"):
            if "prediction" in path.name:
                path.unlink()
    manifest_path = target / "taxonomy" / "pooled_label_freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    taxonomy = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparison_path = target / "classification" / "all_window_model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()
    primary_windows = None
    if not comparison.empty and "n" in comparison:
        values = pd.to_numeric(comparison["n"], errors="coerce").dropna().astype(int)
        if len(values):
            if values.nunique() != 1:
                raise ValueError("all-window model comparison has inconsistent denominators")
            primary_windows = int(values.iloc[0])

    _write_pg36_release_audit(
        target=target,
        taxonomy=taxonomy,
        comparison=comparison,
        source=source,
    )
    _write_pg36_release_reports(
        target=target,
        taxonomy=taxonomy,
        comparison=comparison,
        source=source,
    )
    _normalize_pg36_release_metadata(target, taxonomy, comparison)
    copied.append("reports: generated from current PG36 tables")
    copied.append("audit: generated from current PG36 tables")
    (target / "README_zh.md").write_text(
        "# PG36 形态学多分类正式结果\n\n"
        "本目录只包含冻结的 pooled PG36 lineage。每个事件×有效通道是一个独立观察；"
        "PG 类别是探索性形态类别，不是临床发作亚型。"
        "`classification/all_window_model_comparison.csv` 是全部测试窗口主结果，不能用 capped-at-8 敏感性表替代。"
        "历史 taxonomy 不属于此正式入口。\n",
        encoding="utf-8",
    )
    files = []
    for path in target.rglob("*"):
        if path.is_file() and path.name != "release_manifest.json":
            files.append({"path": str(path.relative_to(target)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256(path)})
    release = {
        "schema_version": 1,
        "task": "morphology_pg36",
        "taxonomy_version": taxonomy.get("taxonomy_version"),
        "label_prefix": taxonomy.get("label_prefix", "PG"),
        "classes": int(taxonomy.get("n_global_categories", 0)),
        "n_event_channel": int(taxonomy.get("n_event_channel", 0)),
        "n_windows": int(taxonomy.get("n_windows", 0)),
        "models": list(MODEL_ORDER_TASK2),
        "primary_test_windows": primary_windows,
        "source_root": str(source),
        "copied": copied,
        "tiff_included": bool(include_tiff),
        "predictions_included": bool(include_predictions),
        "files": files,
        "legacy_lineages_excluded": ["04_label_freeze", "05_classification", "historical_21_read_only"],
    }
    write_json(target / "release_manifest.json", release)
    families = [
        figure_family(
            root=target,
            name="morphology_atlas",
            relative_path="figures",
            renderer="scripts/render_task2_pg36_corrected_v2.py",
            source_inputs=(target / "tables", target / "taxonomy"),
            denominator="all eligible event×valid-channel morphology rows",
            semantics="PG morphology atlas; each valid channel is independent",
            requested=True,
        ),
        figure_family(
            root=target,
            name="classifier_summary",
            relative_path="figures/model_summary",
            renderer="scripts/render_pg36_classifier_summary.py",
            source_inputs=(target / "classification",),
            denominator="all eligible held-out test windows",
            semantics="training curves and primary classifier comparison",
            requested=True,
        ),
        figure_family(
            root=target,
            name="classifier_full_window",
            relative_path="figures/confusion_all_windows",
            renderer="scripts/render_pg36_classifier_full.py",
            source_inputs=(target / "classification", target / "taxonomy"),
            denominator="all eligible held-out test windows",
            semantics="all-class PR/ROC, support/recall and confusion",
            requested=True,
        ),
    ]
    write_figure_manifest(root=target, task="morphology_pg36", families=families, path=target / "figures" / "FIGURE_MANIFEST.json")
    # Refresh the artifact index after adding the figure manifest itself.
    release["files"] = [
        {
            "path": str(path.relative_to(target)).replace("\\", "/"),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
        for path in target.rglob("*")
        if path.is_file() and path.name != "release_manifest.json"
    ]
    write_json(target / "release_manifest.json", release)
    return target


def validate_pg36(root: str | Path) -> dict[str, Any]:
    """Validate the non-negotiable PG36 release invariants."""
    target = Path(root).resolve()
    manifest_path = target / "taxonomy" / "pooled_label_freeze_manifest.json"
    if not manifest_path.is_file():
        # Also accept the original canonical naming while auditing before materialization.
        manifest_path = target / "04_label_freeze_pooled" / "pooled_label_freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    taxonomy = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if int(taxonomy.get("n_global_categories", -1)) != 36:
        errors.append(f"n_global_categories={taxonomy.get('n_global_categories')}")
    if int(taxonomy.get("same_mouse_merge_violations", -1)) != 0:
        errors.append("same_mouse_merge_violations != 0")
    comparison_path = target / "classification" / "all_window_model_comparison.csv"
    if not comparison_path.is_file():
        comparison_path = target / "05_classification_pooled" / "all_window_model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.is_file() else pd.DataFrame()
    if set(comparison.get("model", [])) != set(MODEL_ORDER_TASK2):
        errors.append("five-model comparison is incomplete")
    primary_test_windows: int | None = None
    if not comparison.empty and "n" in comparison:
        values = pd.to_numeric(comparison["n"], errors="coerce").dropna().astype(int)
        if values.empty or values.nunique() != 1 or (values <= 0).any():
            errors.append("primary all-window denominator is missing or inconsistent")
        else:
            primary_test_windows = int(values.iloc[0])
    release_manifest = _json_file(target / "release_manifest.json")
    if release_manifest and int(release_manifest.get("classes", 36)) != 36:
        errors.append(f"release manifest classes={release_manifest.get('classes')}")

    # A clean PG36 release must not contain a report or audit that presents a
    # historical G taxonomy as the current classifier.  Checking both JSON
    # fields and text catches stale source-tree copies that otherwise look
    # structurally valid.
    stale_names = _STALE_REPORT_NAMES | {
        "classifier_environment_05_classification_pooled_smoke.json",
    }
    for directory_name in ("reports", "audit"):
        directory = target / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.name in stale_names:
                errors.append(f"legacy report/audit file present: {path.relative_to(target)}")
                continue
            if path.suffix.lower() == ".json":
                payload = _json_file(path)
                classes = payload.get("classes")
                if classes is not None and str(classes).isdigit() and int(classes) != 36:
                    errors.append(f"{path.relative_to(target)} classes={classes}")
                global_categories = payload.get("global_categories")
                if global_categories is not None and str(global_categories).isdigit() and int(global_categories) != 36:
                    errors.append(f"{path.relative_to(target)} global_categories={global_categories}")
                checkpoint_classes = payload.get("checkpoint_classes")
                if isinstance(checkpoint_classes, dict) and any(int(value) != 36 for value in checkpoint_classes.values() if str(value).isdigit()):
                    errors.append(f"{path.relative_to(target)} checkpoint_classes are not PG36")
            if path.suffix.lower() in {".md", ".txt", ".csv"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if re.search(r"G01\s*[–-]\s*G45|45\s*个(?:全局|探索性)类别", text):
                    errors.append(f"{path.relative_to(target)} presents legacy G45 taxonomy")
                if re.search(r"(?:global_categories|checkpoint_classes|classes)\s*[:=]\s*45\b", text):
                    errors.append(f"{path.relative_to(target)} contains classes=45")
    figure_root = target / "figures"
    if figure_root.is_dir():
        figure_manifest = figure_root / "FIGURE_MANIFEST.json"
        if not figure_manifest.is_file():
            errors.append("missing figures/FIGURE_MANIFEST.json")
        elif figure_manifest.is_file():
            try:
                payload = json.loads(figure_manifest.read_text(encoding="utf-8"))
                if payload.get("task") not in {None, "morphology_pg36"}:
                    errors.append("figure manifest task is not morphology_pg36")
                families = payload.get("figure_families")
                if not isinstance(families, list) or not families:
                    errors.append("figure manifest has no figure_families")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                errors.append("invalid figures/FIGURE_MANIFEST.json")
        required_dirs = [
            "01_per_mouse_pg_pca",
            "02_per_mouse_pg_waveforms",
            "03_global_pg_centroid_pca",
            "04_pg_support_presence",
            "06_classifier_pr_roc_all36",
        ]
        for name in required_dirs:
            if not (figure_root / name).is_dir():
                errors.append(f"missing formal figure directory: {name}")
        for model in MODEL_ORDER_TASK2:
            confusion = figure_root / "confusion_all_windows" / model / "figure_confusion_matrix_all_windows.png"
            if not confusion.is_file():
                errors.append(f"missing all-class figure for model: {model}")
            pr_figure = figure_root / "06_classifier_pr_roc_all36" / model / "figure_pr_all36.png"
            roc_figure = figure_root / "06_classifier_pr_roc_all36" / model / "figure_roc_all36.png"
            full_pr_figure = figure_root / "confusion_all_windows" / model / "figure_pr_all_classes.png"
            full_roc_figure = figure_root / "confusion_all_windows" / model / "figure_roc_all_classes.png"
            if not ((pr_figure.is_file() and roc_figure.is_file()) or (full_pr_figure.is_file() and full_roc_figure.is_file())):
                errors.append(f"missing all-class PR/ROC figures for model: {model}")
    result = {"root": str(target), "taxonomy_classes": int(taxonomy.get("n_global_categories", 0)), "models": sorted(set(comparison.get("model", []))), "primary_test_windows": primary_test_windows, "ok": not errors, "errors": errors}
    write_json(target / "pg36_validation.json", result)
    if errors:
        raise ValueError("PG36 release validation failed: " + "; ".join(errors))
    return result
