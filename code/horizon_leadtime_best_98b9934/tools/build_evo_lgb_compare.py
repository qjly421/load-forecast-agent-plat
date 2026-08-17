#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese markdown compare record for Shandong evo LGB runs.")
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--compare-id", type=str, required=True)
    parser.add_argument("--issue-id", type=str, default="reference-only")
    parser.add_argument("--line-id", type=str, default="reference-line")
    parser.add_argument("--baseline-run-id", type=str, default=None)
    parser.add_argument("--candidate-run-id", type=str, default=None)
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--candidate-label", type=str, default="candidate")
    parser.add_argument("--decision-status", type=str, default="reference_only")
    parser.add_argument("--record-md-relpath", type=str, default=None)
    parser.add_argument("--notes", action="append", default=[])
    parser.add_argument("--control-only", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _format_float(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def _format_delta(candidate: float | None, baseline: float | None) -> str:
    if candidate is None or baseline is None or pd.isna(candidate) or pd.isna(baseline):
        return ""
    delta = float(candidate) - float(baseline)
    return f"{delta:+.4f}"


def _segment_value(summary: dict, label: str) -> float | None:
    for item in summary.get("segment_summaries", []):
        if str(item.get("label")) == label:
            value = item.get("overall_acc")
            return None if value is None else float(value)
    return None


def _summarize_numeric(values: list[float]) -> tuple[float | None, str]:
    if not values:
        return None, "NA"
    if len(values) == 1:
        value = float(values[0])
        return value, _format_float(value)
    mean_value = float(sum(values) / len(values))
    text = (
        f"mean={mean_value:.4f} "
        f"min={min(values):.4f} "
        f"max={max(values):.4f} "
        f"n={len(values)}"
    )
    return mean_value, text


def _run_text(meta: dict, train_logs: list[dict]) -> tuple[str, str]:
    train_rows = [int(item["n_train_rows"]) for item in train_logs if item.get("n_train_rows") is not None]
    valid_rows = [int(item["n_valid_rows"]) for item in train_logs if item.get("n_valid_rows") is not None]
    if not train_rows and not valid_rows:
        return "NA", "NA"
    if len(train_rows) <= 1 and len(valid_rows) <= 1:
        return str(train_rows[0] if train_rows else "NA"), str(valid_rows[0] if valid_rows else "NA")
    train_text = f"mean={sum(train_rows)/len(train_rows):.1f} min={min(train_rows)} max={max(train_rows)} n={len(train_rows)}" if train_rows else "NA"
    valid_text = f"mean={sum(valid_rows)/len(valid_rows):.1f} min={min(valid_rows)} max={max(valid_rows)} n={len(valid_rows)}" if valid_rows else "NA"
    return train_text, valid_text


def load_run_context(run_dir: Path, run_id_override: str | None, label: str) -> dict:
    summary = _read_yaml(run_dir / "formal_summary.yaml")
    horizon = pd.read_csv(run_dir / "horizon_summary.csv")
    target_day_summary = pd.read_csv(run_dir / "target_day_summary.csv")
    target_day_horizon = pd.read_csv(run_dir / "target_day_horizon_metrics.csv")
    run_index = pd.read_csv(run_dir / "run_index.csv")
    metadata = _read_json(run_dir / "metadata.json")

    d1 = float(horizon.loc[horizon["dayplus"] == 1, "overall_acc"].iloc[0]) if not horizon.empty else None
    d14 = float(horizon.loc[horizon["dayplus"] == 14, "overall_acc"].iloc[0]) if not horizon.empty else None
    worst_day_row = target_day_summary.sort_values("overall_acc", ascending=True).iloc[0] if not target_day_summary.empty else None
    worst_cell_row = target_day_horizon.sort_values("overall_acc", ascending=True).iloc[0] if not target_day_horizon.empty else None

    train_logs: list[dict] = []
    if "train_log_json" in run_index.columns:
        for raw in run_index["train_log_json"].dropna().astype(str):
            train_logs.extend(json.loads(raw))
    valid_accs = [float(item["valid_acc"]) for item in train_logs if item.get("valid_acc") is not None]
    valid_acc_value, valid_acc_text = _summarize_numeric(valid_accs)
    valid_formal_gap = None if valid_acc_value is None else float(valid_acc_value - float(summary["overall_acc"]))
    train_rows_text, valid_rows_text = _run_text(metadata, train_logs)

    return {
        "label": label,
        "run_id": run_id_override or metadata.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "archive_dir": metadata.get("archive_effective_dir") or metadata.get("archive_dir"),
        "branch": (metadata.get("git") or {}).get("branch"),
        "commit": (metadata.get("git") or {}).get("commit"),
        "overall_acc": float(summary["overall_acc"]),
        "mid_acc": float(summary["mid_acc"]),
        "night_acc": float(summary["night_acc"]),
        "D1": d1,
        "D14": d14,
        "D1-D5": _segment_value(summary, "D1-D5"),
        "D6-D10": _segment_value(summary, "D6-D10"),
        "D11-D14": _segment_value(summary, "D11-D14"),
        "D1->D14 drop": None if d1 is None or d14 is None else float(d1 - d14),
        "worst target_day": None
        if worst_day_row is None
        else f"{worst_day_row['target_day']} ({float(worst_day_row['overall_acc']):.4f})",
        "worst target_day value": None if worst_day_row is None else float(worst_day_row["overall_acc"]),
        "worst target_day × horizon": None
        if worst_cell_row is None
        else f"{worst_cell_row['target_day']} × D{int(worst_cell_row['dayplus'])} ({float(worst_cell_row['overall_acc']):.4f})",
        "worst target_day × horizon value": None if worst_cell_row is None else float(worst_cell_row["overall_acc"]),
        "valid_acc": valid_acc_value,
        "valid_acc_text": valid_acc_text,
        "valid_formal_gap": valid_formal_gap,
        "train_rows_text": train_rows_text,
        "valid_rows_text": valid_rows_text,
    }


def build_table_rows(baseline: dict, candidate: dict) -> list[tuple[str, str, str, str]]:
    metrics = [
        "overall_acc",
        "mid_acc",
        "night_acc",
        "D1",
        "D14",
        "D1-D5",
        "D6-D10",
        "D11-D14",
        "D1->D14 drop",
        "valid_acc",
        "valid_formal_gap",
    ]
    rows: list[tuple[str, str, str, str]] = []
    for key in metrics:
        base_value = baseline.get(key)
        cand_value = candidate.get(key)
        rows.append((key, _format_float(base_value), _format_float(cand_value), _format_delta(cand_value, base_value)))

    rows.extend(
        [
            ("worst target_day", str(baseline.get("worst target_day", "NA")), str(candidate.get("worst target_day", "NA")), ""),
            (
                "worst target_day × horizon",
                str(baseline.get("worst target_day × horizon", "NA")),
                str(candidate.get("worst target_day × horizon", "NA")),
                "",
            ),
            ("train_rows", baseline.get("train_rows_text", "NA"), candidate.get("train_rows_text", "NA"), ""),
            ("valid_rows", baseline.get("valid_rows_text", "NA"), candidate.get("valid_rows_text", "NA"), ""),
            ("result_dir", baseline["run_dir"], candidate["run_dir"], ""),
            ("archive_dir", str(baseline.get("archive_dir")), str(candidate.get("archive_dir")), ""),
            ("branch", str(baseline.get("branch")), str(candidate.get("branch")), ""),
            ("commit", str(baseline.get("commit")), str(candidate.get("commit")), ""),
        ]
    )
    return rows


def build_markdown(*, args: argparse.Namespace, baseline: dict, candidate: dict) -> str:
    rows = build_table_rows(baseline, candidate)
    notes = args.notes or []
    note_lines = "\n".join(f"- {item}" for item in notes) if notes else "- 无"
    table_lines = "\n".join(
        f"| {metric} | {base} | {cand} | {delta} |" for metric, base, cand, delta in rows
    )
    auto_lines = [
        f"- `overall_acc` 差值：{_format_delta(candidate['overall_acc'], baseline['overall_acc'])}",
        f"- `D11-D14` 差值：{_format_delta(candidate['D11-D14'], baseline['D11-D14'])}",
        f"- `D1->D14 drop` 差值：{_format_delta(candidate['D1->D14 drop'], baseline['D1->D14 drop'])}",
        f"- `valid_formal_gap` 差值：{_format_delta(candidate['valid_formal_gap'], baseline['valid_formal_gap'])}",
    ]
    auto_lines_text = "\n".join(auto_lines)
    return f"""# {args.compare_id}

## 元信息

- compare_id: `{args.compare_id}`
- issue_id: `{args.issue_id}`
- line_id: `{args.line_id}`
- baseline_run_id: `{baseline['run_id']}`
- candidate_run_id: `{candidate['run_id']}`
- decision_status: `{args.decision_status}`
- control_only: `{str(bool(args.control_only)).lower()}`
- record_md_relpath: `{args.record_md_relpath or args.output_md.as_posix()}`
- generated_at: `{datetime.now().isoformat()}`

## 对比对象

- baseline: `{args.baseline_label}` -> `{baseline['run_dir']}`
- candidate: `{args.candidate_label}` -> `{candidate['run_dir']}`

## 指标对比

| metric | baseline | candidate | delta |
| --- | --- | --- | --- |
{table_lines}

## 自动观察

{auto_lines_text}

## 备注

{note_lines}
"""


def main() -> None:
    args = parse_args()
    baseline = load_run_context(args.baseline_run_dir, args.baseline_run_id, args.baseline_label)
    candidate = load_run_context(args.candidate_run_dir, args.candidate_run_id, args.candidate_label)
    markdown = build_markdown(args=args, baseline=baseline, candidate=candidate)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
