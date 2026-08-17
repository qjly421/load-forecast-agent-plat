#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from evaluator import (  # noqa: E402
    build_formal_summary,
    build_horizon_summary,
    build_per_point_summary,
    build_target_day_summary,
)


DEFAULT_MANIFEST = PACKAGE_ROOT / "config" / "audit" / "current_formal_runs.yaml"
BASE_REQUIRED_FILES = [
    "plan.json",
    "run_index.csv",
    "predictions_long.csv",
    "target_day_horizon_metrics.csv",
    "horizon_summary.csv",
    "target_day_summary.csv",
    "formal_summary.yaml",
    "metadata.json",
]
PREDICTION_REQUIRED_COLUMNS = [
    "benchmark_point",
    "stage",
    "model_family",
    "nn_backbone_id",
    "strategy_id",
    "train_group_id",
    "machine",
    "gpu_slot",
    "target_day",
    "origin_day",
    "dayplus",
    "timestamp",
    "actual",
    "pred",
]
SUMMARY_METRIC_KEYS = ["overall_acc", "mid_acc", "night_acc", "mae", "mse", "rmse"]
SUMMARY_TOP_LEVEL_KEYS = [
    "profile_id",
    "stage",
    "model_family",
    "nn_backbone_id",
    "backbone_type",
    "source_native_config_template",
    "normalization_method",
    "time_embedding_type",
    "only_output_after_gap",
    "gap_steps",
    "day2_weight",
    "midday_penalty",
    "strategy_id",
    "seed",
    "n_target_days",
    *SUMMARY_METRIC_KEYS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit archived Shandong target-day formal benchmark runs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--artifact-mode", choices=["archive", "machine"], default="archive")
    parser.add_argument(
        "--machine-filter",
        action="append",
        default=[],
        help="Only audit runs whose manifest machine matches one of these values. Repeatable.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def scalar_equal(left: Any, right: Any, *, tol: float = 1e-6) -> bool:
    if is_missing(left) and is_missing(right):
        return True
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol)
        except Exception:
            return False
    return str(left) == str(right)


def compare_scalar(*, context: str, key: str, actual: Any, expected: Any, failures: list[str], tol: float = 1e-6) -> None:
    if not scalar_equal(actual, expected, tol=tol):
        failures.append(f"{context}: {key} mismatch, actual={actual!r}, expected={expected!r}")


def compare_frame(
    *,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
    context: str,
    failures: list[str],
    tol: float = 1e-6,
) -> None:
    actual = actual.copy()
    expected = expected.copy()
    for frame in (actual, expected):
        for col in key_cols:
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
    actual = actual.sort_values(key_cols).reset_index(drop=True)
    expected = expected.sort_values(key_cols).reset_index(drop=True)
    if list(actual.columns) != list(expected.columns):
        failures.append(f"{context}: columns mismatch, actual={list(actual.columns)}, expected={list(expected.columns)}")
        return
    if len(actual) != len(expected):
        failures.append(f"{context}: row count mismatch, actual={len(actual)}, expected={len(expected)}")
        return
    merged = actual.merge(expected, how="outer", on=key_cols, suffixes=("_actual", "_expected"), indicator=True)
    only_one_side = merged.loc[merged["_merge"] != "both"]
    if not only_one_side.empty:
        sample = only_one_side.head(10).to_dict(orient="records")
        failures.append(f"{context}: key rows mismatch, sample={sample}")
        return
    for row in merged.itertuples(index=False):
        for col in value_cols:
            actual_value = getattr(row, f"{col}_actual")
            expected_value = getattr(row, f"{col}_expected")
            if not scalar_equal(actual_value, expected_value, tol=tol):
                key_payload = {col_name: getattr(row, col_name) for col_name in key_cols}
                failures.append(
                    f"{context}: value mismatch at {key_payload}, column={col}, actual={actual_value!r}, expected={expected_value!r}"
                )
                return


def point_target_days(summary_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(point): int(len(group))
        for point, group in summary_df.groupby("benchmark_point", sort=True)
    }


def point_origin_calls(run_index_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(point): int(group["n_assigned_origins"].astype(int).sum())
        for point, group in run_index_df.groupby("benchmark_point", sort=True)
    }


def point_data_max_timestamp(metadata: dict[str, Any]) -> str | None:
    data_summary = metadata.get("data_summary") or {}
    end_value = data_summary.get("end")
    return None if end_value is None else str(end_value)


def load_profile_points(profile_path: Path) -> list[dict[str, Any]]:
    profile = read_yaml(profile_path)
    return list(profile.get("points", []))


def audit_run(*, entry: dict[str, Any], archive_root: Path, artifact_mode: str) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if artifact_mode == "archive":
        run_dir = archive_root / entry["archive_subdir"]
    elif artifact_mode == "machine":
        run_dir = Path(entry["machine_result_dir"])
    else:
        raise ValueError(f"unsupported artifact_mode: {artifact_mode}")
    if not run_dir.exists():
        failures.append(f"result directory missing: {run_dir}")
        return {"run_id": entry["run_id"], "status": "failed", "run_dir": str(run_dir), "failures": failures, "warnings": warnings}

    required_files = list(BASE_REQUIRED_FILES)
    if entry.get("expect_per_point_file"):
        required_files.append("per_point_summary.csv")
    missing_files = [name for name in required_files if not (run_dir / name).exists()]
    if missing_files:
        failures.append(f"missing required files: {missing_files}")
        return {"run_id": entry["run_id"], "status": "failed", "run_dir": str(run_dir), "failures": failures, "warnings": warnings}

    plan = read_json(run_dir / "plan.json")
    metadata = read_json(run_dir / "metadata.json")
    formal_summary = read_yaml(run_dir / "formal_summary.yaml")
    run_index = pd.read_csv(run_dir / "run_index.csv")
    predictions = pd.read_csv(run_dir / "predictions_long.csv")
    target_horizon = pd.read_csv(run_dir / "target_day_horizon_metrics.csv")
    horizon_summary = pd.read_csv(run_dir / "horizon_summary.csv")
    target_day_summary = pd.read_csv(run_dir / "target_day_summary.csv")
    per_point_summary = pd.read_csv(run_dir / "per_point_summary.csv") if (run_dir / "per_point_summary.csv").exists() else pd.DataFrame()

    compare_scalar(context=entry["run_id"], key="metadata.profile_id", actual=metadata.get("profile_id"), expected=entry["profile_id"], failures=failures)
    compare_scalar(context=entry["run_id"], key="metadata.stage", actual=metadata.get("stage"), expected=entry["stage"], failures=failures)
    compare_scalar(context=entry["run_id"], key="metadata.model_family", actual=metadata.get("model_family"), expected=entry["model_family"], failures=failures)
    compare_scalar(context=entry["run_id"], key="metadata.machine", actual=metadata.get("machine"), expected=entry["machine"], failures=failures)
    compare_scalar(context=entry["run_id"], key="formal_summary.strategy_id", actual=formal_summary.get("strategy_id"), expected=entry["strategy_id"], failures=failures)
    compare_scalar(context=entry["run_id"], key="git.branch", actual=(metadata.get("git") or {}).get("branch"), expected=entry["branch"], failures=failures)
    compare_scalar(context=entry["run_id"], key="git.commit", actual=(metadata.get("git") or {}).get("commit"), expected=entry["commit"], failures=failures)
    compare_scalar(context=entry["run_id"], key="run_index_rows", actual=len(run_index), expected=entry["expected_run_index_rows"], failures=failures)

    if not run_index["status"].eq("success").all():
        bad_rows = run_index.loc[run_index["status"] != "success"].to_dict(orient="records")
        failures.append(f"{entry['run_id']}: non-success run_index rows detected: {bad_rows[:5]}")

    missing_prediction_columns = sorted(set(PREDICTION_REQUIRED_COLUMNS) - set(predictions.columns))
    if missing_prediction_columns:
        failures.append(f"{entry['run_id']}: predictions_long missing columns {missing_prediction_columns}")

    combo_counts = (
        predictions.groupby(["benchmark_point", "target_day", "dayplus"], sort=True)
        .size()
        .reset_index(name="n_rows")
    )
    bad_combo_counts = combo_counts.loc[combo_counts["n_rows"] != 96]
    if not bad_combo_counts.empty:
        failures.append(
            f"{entry['run_id']}: predictions_long has non-96 target_day/dayplus rows, sample={bad_combo_counts.head(10).to_dict(orient='records')}"
        )

    if not target_day_summary["n_horizons"].astype(int).eq(14).all():
        failures.append(f"{entry['run_id']}: target_day_summary contains n_horizons != 14")
    if not target_horizon["n_points"].astype(int).eq(96).all():
        failures.append(f"{entry['run_id']}: target_day_horizon_metrics contains n_points != 96")

    actual_target_days = point_target_days(target_day_summary)
    expected_target_days = {str(k): int(v) for k, v in (entry.get("expected_target_days_by_point") or {}).items()}
    if actual_target_days != expected_target_days:
        failures.append(f"{entry['run_id']}: target-day coverage mismatch, actual={actual_target_days}, expected={expected_target_days}")

    actual_origin_calls = point_origin_calls(run_index)
    expected_origin_calls = {str(k): int(v) for k, v in (entry.get("expected_origin_calls_by_point") or {}).items()}
    if actual_origin_calls != expected_origin_calls:
        failures.append(f"{entry['run_id']}: origin-call coverage mismatch, actual={actual_origin_calls}, expected={expected_origin_calls}")

    recalculated_horizon_summary = build_horizon_summary(target_horizon)
    recalculated_target_day_summary = build_target_day_summary(target_horizon)
    recalculated_per_point_summary = build_per_point_summary(recalculated_target_day_summary)
    recalculated_formal_summary = build_formal_summary(
        profile_id=str(metadata["profile_id"]),
        stage=str(metadata["stage"]),
        model_family=str(metadata["model_family"]),
        strategy_id=str(formal_summary["strategy_id"]),
        seed=int(metadata["seed"]),
        target_horizon_df=target_horizon,
        horizon_summary=recalculated_horizon_summary,
        target_day_summary=recalculated_target_day_summary,
        nn_backbone_id=metadata.get("nn_backbone_id"),
        backbone_type=metadata.get("backbone_type"),
        source_native_config_template=metadata.get("source_native_config_template"),
        native_model_contract=metadata.get("native_model_contract"),
    )
    if not recalculated_per_point_summary.empty:
        recalculated_formal_summary["per_point_summary"] = recalculated_per_point_summary.to_dict(orient="records")

    compare_frame(
        actual=horizon_summary,
        expected=recalculated_horizon_summary,
        key_cols=["dayplus"],
        value_cols=["label", "n_target_days", *SUMMARY_METRIC_KEYS],
        context=f"{entry['run_id']}: horizon_summary",
        failures=failures,
    )
    compare_frame(
        actual=target_day_summary,
        expected=recalculated_target_day_summary,
        key_cols=["benchmark_point", "target_day"],
        value_cols=["n_horizons", *SUMMARY_METRIC_KEYS],
        context=f"{entry['run_id']}: target_day_summary",
        failures=failures,
    )
    if entry.get("expect_per_point_file"):
        compare_frame(
            actual=per_point_summary,
            expected=recalculated_per_point_summary,
            key_cols=["benchmark_point"],
            value_cols=["n_target_days", *SUMMARY_METRIC_KEYS],
            context=f"{entry['run_id']}: per_point_summary",
            failures=failures,
        )
    elif not per_point_summary.empty:
        warnings.append(f"{entry['run_id']}: unexpected per_point_summary.csv exists for non-stage4 run")

    for key in SUMMARY_TOP_LEVEL_KEYS:
        compare_scalar(
            context=f"{entry['run_id']}: formal_summary",
            key=key,
            actual=formal_summary.get(key),
            expected=recalculated_formal_summary.get(key),
            failures=failures,
        )

    expected_segments = {item["label"]: item for item in recalculated_formal_summary.get("segment_summaries", [])}
    actual_segments = {item["label"]: item for item in formal_summary.get("segment_summaries", [])}
    if set(actual_segments) != set(expected_segments):
        failures.append(
            f"{entry['run_id']}: formal_summary.segment_summaries labels mismatch, actual={sorted(actual_segments)}, expected={sorted(expected_segments)}"
        )
    else:
        for label, expected_row in expected_segments.items():
            actual_row = actual_segments[label]
            for key in SUMMARY_METRIC_KEYS:
                compare_scalar(
                    context=f"{entry['run_id']}: segment_summaries[{label}]",
                    key=key,
                    actual=actual_row.get(key),
                    expected=expected_row.get(key),
                    failures=failures,
                )

    expected_horizon_rows = pd.DataFrame(recalculated_formal_summary.get("horizon_rows", []))
    actual_horizon_rows = pd.DataFrame(formal_summary.get("horizon_rows", []))
    if not expected_horizon_rows.empty or not actual_horizon_rows.empty:
        compare_frame(
            actual=actual_horizon_rows,
            expected=expected_horizon_rows,
            key_cols=["dayplus"],
            value_cols=["label", "overall_acc", "mid_acc", "night_acc"],
            context=f"{entry['run_id']}: formal_summary.horizon_rows",
            failures=failures,
        )

    for nested_key in ["worst_target_day", "worst_horizon", "worst_target_day_horizon"]:
        actual_nested = formal_summary.get(nested_key) or {}
        expected_nested = recalculated_formal_summary.get(nested_key) or {}
        if set(actual_nested.keys()) != set(expected_nested.keys()):
            failures.append(
                f"{entry['run_id']}: {nested_key} keys mismatch, actual={sorted(actual_nested.keys())}, expected={sorted(expected_nested.keys())}"
            )
            continue
        for key in expected_nested:
            compare_scalar(
                context=f"{entry['run_id']}: {nested_key}",
                key=key,
                actual=actual_nested.get(key),
                expected=expected_nested.get(key),
                failures=failures,
            )

    if "per_point_summary" in formal_summary:
        compare_frame(
            actual=pd.DataFrame(formal_summary["per_point_summary"]),
            expected=recalculated_per_point_summary,
            key_cols=["benchmark_point"],
            value_cols=["n_target_days", *SUMMARY_METRIC_KEYS],
            context=f"{entry['run_id']}: formal_summary.per_point_summary",
            failures=failures,
        )

    headline = {key: formal_summary.get(key) for key in ["overall_acc", "mid_acc", "night_acc", "mae", "rmse", "n_target_days"]}
    status = "passed" if not failures else "failed"
    return {
        "run_id": entry["run_id"],
        "status": status,
        "run_dir": str(run_dir),
        "headline": headline,
        "failures": failures,
        "warnings": warnings,
        "data_max_timestamp": point_data_max_timestamp(metadata),
    }


def audit_blocker(
    *,
    entry: dict[str, Any],
    repo_root: Path,
    run_results_by_id: dict[str, dict[str, Any]],
    artifact_mode: str,
    selected_run_ids: set[str],
) -> dict[str, Any]:
    failures: list[str] = []
    if artifact_mode != "archive":
        return {
            "blocker_id": entry["blocker_id"],
            "status": "skipped_artifact_mode",
            "expected_last_complete_target_day": entry["expected_last_complete_target_day"],
            "failures": [],
        }
    package_root = Path(__file__).resolve().parents[1]
    intended_profile = package_root / entry["intended_profile_relpath"]
    executable_profile = package_root / entry["executable_profile_relpath"]
    evidence_doc = package_root / entry["evidence_doc_relpath"]

    intended_points = load_profile_points(intended_profile)
    executable_points = load_profile_points(executable_profile)
    intended_last = intended_points[-1]
    executable_last = executable_points[-1]

    compare_scalar(
        context=entry["blocker_id"],
        key="intended_last_point_id",
        actual=intended_last.get("point_id"),
        expected=entry["expected_ideal_last_point_id"],
        failures=failures,
    )
    compare_scalar(
        context=entry["blocker_id"],
        key="intended_last_point_end",
        actual=intended_last.get("target_end"),
        expected=entry["expected_ideal_last_point_end"],
        failures=failures,
    )
    compare_scalar(
        context=entry["blocker_id"],
        key="executable_last_point_id",
        actual=executable_last.get("point_id"),
        expected=entry["expected_executable_last_point_id"],
        failures=failures,
    )
    compare_scalar(
        context=entry["blocker_id"],
        key="executable_last_point_end",
        actual=executable_last.get("target_end"),
        expected=entry["expected_executable_last_point_end"],
        failures=failures,
    )

    doc_text = evidence_doc.read_text(encoding="utf-8")
    for phrase in entry.get("required_doc_phrases", []):
        if phrase not in doc_text:
            failures.append(f"{entry['blocker_id']}: missing evidence phrase in doc: {phrase}")

    missing_selected = [run_id for run_id in entry.get("evidence_run_ids", []) if run_id not in selected_run_ids]
    if missing_selected:
        return {
            "blocker_id": entry["blocker_id"],
            "status": "skipped_subset",
            "expected_last_complete_target_day": entry["expected_last_complete_target_day"],
            "failures": [],
            "skipped_missing_runs": missing_selected,
        }

    for run_id in entry.get("evidence_run_ids", []):
        result = run_results_by_id.get(run_id)
        if result is None:
            failures.append(f"{entry['blocker_id']}: evidence run {run_id} missing from formal run audit results")
            continue
        if result.get("data_max_timestamp") is None:
            failures.append(f"{entry['blocker_id']}: evidence run {run_id} does not expose data_max_timestamp")
            continue
        compare_scalar(
            context=f"{entry['blocker_id']}:{run_id}",
            key="data_max_timestamp",
            actual=result.get("data_max_timestamp"),
            expected=entry["expected_data_max_timestamp"],
            failures=failures,
        )

    status = "verified_blocked" if not failures else "failed"
    return {
        "blocker_id": entry["blocker_id"],
        "status": status,
        "expected_last_complete_target_day": entry["expected_last_complete_target_day"],
        "failures": failures,
    }


def build_report(
    *,
    manifest_path: Path,
    archive_root: Path,
    repo_root: Path,
    artifact_mode: str,
    machine_filters: list[str],
) -> dict[str, Any]:
    manifest = read_yaml(manifest_path)
    selected_entries = []
    machine_filter_set = {item.strip() for item in machine_filters if item and item.strip()}
    for entry in manifest.get("formal_runs", []):
        if machine_filter_set and str(entry.get("machine")) not in machine_filter_set:
            continue
        selected_entries.append(entry)
    run_results = [audit_run(entry=entry, archive_root=archive_root, artifact_mode=artifact_mode) for entry in selected_entries]
    run_results_by_id = {item["run_id"]: item for item in run_results}
    selected_run_ids = set(run_results_by_id)
    blocker_results = [
        audit_blocker(
            entry=entry,
            repo_root=repo_root,
            run_results_by_id=run_results_by_id,
            artifact_mode=artifact_mode,
            selected_run_ids=selected_run_ids,
        )
        for entry in manifest.get("blockers", [])
    ]

    failed_runs = [item["run_id"] for item in run_results if item["status"] != "passed"]
    failed_blockers = [item["blocker_id"] for item in blocker_results if item["status"] == "failed"]
    return {
        "manifest_path": str(manifest_path),
        "archive_root": str(archive_root),
        "repo_root": str(repo_root),
        "artifact_mode": artifact_mode,
        "machine_filters": sorted(machine_filter_set),
        "current_formal_baseline": manifest.get("current_formal_baseline", {}),
        "formal_runs": run_results,
        "blockers": blocker_results,
        "status": "passed" if not failed_runs and not failed_blockers else "failed",
        "failed_runs": failed_runs,
        "failed_blockers": failed_blockers,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"manifest: {report['manifest_path']}")
    print(f"archive_root: {report['archive_root']}")
    print(f"artifact_mode: {report['artifact_mode']}")
    if report.get("machine_filters"):
        print(f"machine_filters: {', '.join(report['machine_filters'])}")
    baseline = report.get("current_formal_baseline") or {}
    if baseline:
        print(
            "current_formal_baseline: "
            f"{baseline.get('stage')} / {baseline.get('profile_id')} "
            f"({baseline.get('note')})"
        )
    print("")
    print("Formal Runs")
    for item in report["formal_runs"]:
        headline = item.get("headline") or {}
        print(
            f"- {item['run_id']}: {item['status']} "
            f"(overall={headline.get('overall_acc')}, mid={headline.get('mid_acc')}, "
            f"night={headline.get('night_acc')}, n_target_days={headline.get('n_target_days')})"
        )
        print(f"  archive_run_dir: {item['run_dir']}")
        for warning in item.get("warnings", []):
            print(f"  warning: {warning}")
        for failure in item.get("failures", []):
            print(f"  failure: {failure}")
    print("")
    print("Blockers")
    for item in report["blockers"]:
        print(
            f"- {item['blocker_id']}: {item['status']} "
            f"(last_complete_target_day={item.get('expected_last_complete_target_day')})"
        )
        for failure in item.get("failures", []):
            print(f"  failure: {failure}")
    print("")
    print(f"overall_status: {report['status']}")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = read_yaml(manifest_path)
    archive_root = args.archive_root.resolve() if args.archive_root else Path(manifest["archive_root"]).resolve()
    repo_root = PACKAGE_ROOT.parent
    report = build_report(
        manifest_path=manifest_path,
        archive_root=archive_root,
        repo_root=repo_root,
        artifact_mode=args.artifact_mode,
        machine_filters=args.machine_filter,
    )
    print_report(report)
    if args.print_json:
        print("")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
