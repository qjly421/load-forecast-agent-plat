from __future__ import annotations

import copy
import json
import os
import random
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = PACKAGE_ROOT / "config" / "formal" / "formal_sep2025.yaml"
DEFAULT_STAGE4_PROFILE = PACKAGE_ROOT / "config" / "formal" / "formal_allpoints.yaml"
DEFAULT_PATHS_CONFIG = PACKAGE_ROOT / "config" / "local_paths.yaml"
EXAMPLE_PATHS_CONFIG = PACKAGE_ROOT / "config" / "local_paths.example.yaml"
PATH_ENV_KEYS = {
    "load_path": "SHANDONG_14D_LOAD_PATH",
    "weather_root": "SHANDONG_14D_WEATHER_ROOT",
    "weather_source": "SHANDONG_14D_WEATHER_SOURCE",
    "merged_cache_root": "SHANDONG_14D_CACHE_ROOT",
    "result_root": "SHANDONG_14D_RESULT_ROOT",
    "archive_root": "SHANDONG_14D_ARCHIVE_ROOT",
}


@dataclass(frozen=True)
class BenchmarkPoint:
    point_id: str
    target_start: date
    target_end: date
    label: str


@dataclass(frozen=True)
class TrainGroup:
    group_id: str
    assigned_origin_days: tuple[date, ...]
    valid_origin_days: tuple[date, ...]
    train_end_day: date
    strategy_id: str
    segmented: bool

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["assigned_origin_days"] = [day.isoformat() for day in self.assigned_origin_days]
        payload["valid_origin_days"] = [day.isoformat() for day in self.valid_origin_days]
        payload["train_end_day"] = self.train_end_day.isoformat()
        return payload


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = base or PACKAGE_ROOT
    return (root / path).resolve()


def resolved_data_config(profile: dict, machine: str | None = None) -> dict:
    data_cfg = copy.deepcopy(profile.get("data", {}))
    overrides = data_cfg.pop("machine_overrides", {}) or {}
    if not machine:
        return data_cfg
    machine_override = copy.deepcopy(overrides.get(machine, {})) or {}
    if machine_override:
        data_cfg.update(machine_override)
    return data_cfg


def _nonempty(value) -> bool:
    return value is not None and str(value).strip() != ""


def _canonical_paths_config(payload: dict) -> dict:
    payload = payload or {}
    canonical: dict = {}
    for key in ["load_path", "weather_root", "weather_source", "result_root", "archive_root"]:
        if _nonempty(payload.get(key)):
            canonical[key] = payload[key]
    cache_value = payload.get("cache_root", payload.get("merged_cache_root"))
    if _nonempty(cache_value):
        canonical["merged_cache_root"] = cache_value
    return canonical


def load_paths_config(path_arg: str | None) -> tuple[dict, Path | None, str]:
    if path_arg:
        path = resolve_path(path_arg, PACKAGE_ROOT)
        if not path.exists():
            raise FileNotFoundError(f"paths config not found: {path}")
        return _canonical_paths_config(read_yaml(path)), path, "explicit"
    if DEFAULT_PATHS_CONFIG.exists():
        return _canonical_paths_config(read_yaml(DEFAULT_PATHS_CONFIG)), DEFAULT_PATHS_CONFIG, "default"
    return {}, None, "not_found"


def _select_runtime_value(
    *,
    key: str,
    profile_values: dict,
    paths_config: dict,
    cli_overrides: dict,
) -> tuple[str | None, str]:
    value = profile_values.get(key)
    source = "profile"
    if _nonempty(paths_config.get(key)):
        value = paths_config[key]
        source = "paths_config"
    env_key = PATH_ENV_KEYS.get(key)
    if env_key and _nonempty(os.environ.get(env_key)):
        value = os.environ[env_key]
        source = f"env:{env_key}"
    if _nonempty(cli_overrides.get(key)):
        value = cli_overrides[key]
        source = "cli"
    if value is None:
        return None, source
    return str(value), source


def _repo_relative_value(value: str | None) -> str | None:
    if not _nonempty(value):
        return None
    path = Path(str(value))
    return None if path.is_absolute() else str(value)


def _path_status(path: Path | None) -> dict:
    if path is None:
        return {
            "path": None,
            "exists": False,
            "is_file": False,
            "is_dir": False,
            "size_bytes": None,
            "mtime": None,
        }
    exists = path.exists()
    payload = {
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(path.is_file()) if exists else False,
        "is_dir": bool(path.is_dir()) if exists else False,
        "size_bytes": None,
        "mtime": None,
    }
    if exists:
        stat = path.stat()
        payload["size_bytes"] = int(stat.st_size) if path.is_file() else None
        payload["mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    return payload


def _resolve_optional_path(value: str | None) -> Path | None:
    if not _nonempty(value):
        return None
    return resolve_path(str(value), PACKAGE_ROOT)


def resolve_runtime_paths(
    *,
    profile: dict,
    machine: str | None = None,
    paths_config_path: str | None = None,
    cli_overrides: dict | None = None,
) -> tuple[dict, dict]:
    paths_config, loaded_paths_config, paths_config_status = load_paths_config(paths_config_path)
    cli_overrides = _canonical_paths_config(cli_overrides or {})
    data_cfg = resolved_data_config(profile, machine)
    profile_values = {
        "load_path": data_cfg.get("load_path"),
        "weather_root": data_cfg.get("weather_root"),
        "weather_source": data_cfg.get("weather_source", "ec"),
        "merged_cache_root": data_cfg.get("merged_cache_root", data_cfg.get("cache_root")),
        "result_root": profile.get("output", {}).get("result_root"),
        "archive_root": profile.get("contract", {}).get("archive_root"),
    }

    selected_raw: dict[str, str | None] = {}
    path_sources: dict[str, str] = {}
    for key in ["load_path", "weather_root", "weather_source", "merged_cache_root", "result_root", "archive_root"]:
        selected_raw[key], path_sources[key] = _select_runtime_value(
            key=key,
            profile_values=profile_values,
            paths_config=paths_config,
            cli_overrides=cli_overrides,
        )

    resolved_path_objects = {
        "load_path": _resolve_optional_path(selected_raw.get("load_path")),
        "weather_root": _resolve_optional_path(selected_raw.get("weather_root")),
        "cache_root": _resolve_optional_path(selected_raw.get("merged_cache_root")),
        "result_root": _resolve_optional_path(selected_raw.get("result_root")),
        "archive_root": _resolve_optional_path(selected_raw.get("archive_root")),
    }
    weather_source = selected_raw.get("weather_source") or "ec"
    weather_source_root = (
        resolved_path_objects["weather_root"] / weather_source
        if resolved_path_objects["weather_root"] is not None
        else None
    )

    if selected_raw.get("result_root") is not None:
        profile.setdefault("output", {})["result_root"] = selected_raw["result_root"]
    if selected_raw.get("archive_root") is not None:
        profile.setdefault("contract", {})["archive_root"] = selected_raw["archive_root"]

    data_cfg["load_path"] = str(resolved_path_objects["load_path"]) if resolved_path_objects["load_path"] else ""
    data_cfg["weather_root"] = str(resolved_path_objects["weather_root"]) if resolved_path_objects["weather_root"] else ""
    data_cfg["weather_source"] = weather_source
    data_cfg["merged_cache_root"] = str(resolved_path_objects["cache_root"]) if resolved_path_objects["cache_root"] else ""

    raw_path_keys = {
        "load_path": selected_raw.get("load_path"),
        "weather_root": selected_raw.get("weather_root"),
        "cache_root": selected_raw.get("merged_cache_root"),
        "result_root": selected_raw.get("result_root"),
        "archive_root": selected_raw.get("archive_root"),
    }
    resolved_paths = {
        key: str(path) if path is not None else None
        for key, path in resolved_path_objects.items()
    }
    resolved_paths["weather_source"] = weather_source
    resolved_paths["weather_source_root"] = str(weather_source_root) if weather_source_root is not None else None

    repo_relative_paths = {
        key: _repo_relative_value(value)
        for key, value in raw_path_keys.items()
    }
    repo_relative_paths["weather_source"] = weather_source

    path_existence = {
        key: _path_status(path)
        for key, path in resolved_path_objects.items()
    }
    path_existence["weather_source_root"] = _path_status(weather_source_root)

    path_sources_for_output = {
        "load_path": path_sources["load_path"],
        "weather_root": path_sources["weather_root"],
        "weather_source": path_sources["weather_source"],
        "cache_root": path_sources["merged_cache_root"],
        "result_root": path_sources["result_root"],
        "archive_root": path_sources["archive_root"],
    }
    raw_selected_paths = {
        key: (str(value) if value is not None else None)
        for key, value in raw_path_keys.items()
    }
    raw_selected_paths["weather_source"] = weather_source

    data_fingerprint = {
        "load_path": resolved_paths["load_path"],
        "load_exists": path_existence["load_path"]["exists"],
        "load_size_bytes": path_existence["load_path"]["size_bytes"],
        "load_mtime": path_existence["load_path"]["mtime"],
        "weather_root": resolved_paths["weather_root"],
        "weather_root_exists": path_existence["weather_root"]["exists"],
        "weather_source": weather_source,
        "weather_source_root": resolved_paths["weather_source_root"],
        "weather_source_root_exists": path_existence["weather_source_root"]["exists"],
    }

    path_context = {
        "priority": "CLI > env > paths_config > profile",
        "paths_config_path": str(loaded_paths_config) if loaded_paths_config is not None else None,
        "paths_config_status": paths_config_status,
        "paths_config_example": str(EXAMPLE_PATHS_CONFIG),
        "resolved_paths": resolved_paths,
        "repo_relative_paths": repo_relative_paths,
        "raw_selected_paths": raw_selected_paths,
        "path_sources": path_sources_for_output,
        "path_existence": path_existence,
        "data_fingerprint": data_fingerprint,
    }
    return data_cfg, path_context


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profile(profile_arg: str | None, *, stage: str | None = None) -> tuple[dict, Path]:
    if profile_arg is None:
        profile_path = DEFAULT_STAGE4_PROFILE if stage == "stage4" else DEFAULT_PROFILE
    else:
        candidate = Path(profile_arg)
        if candidate.suffix in {".yaml", ".yml"}:
            profile_path = resolve_path(candidate, PACKAGE_ROOT)
        else:
            profile_path = PACKAGE_ROOT / "config" / "formal" / f"{profile_arg}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile not found: {profile_path}")
    return read_yaml(profile_path), profile_path


def parse_point(item: dict) -> BenchmarkPoint:
    return BenchmarkPoint(
        point_id=str(item["point_id"]),
        target_start=pd.Timestamp(item["target_start"]).date(),
        target_end=pd.Timestamp(item["target_end"]).date(),
        label=str(item.get("label", item["point_id"])),
    )


def parse_points(profile: dict) -> list[BenchmarkPoint]:
    points = [parse_point(item) for item in profile.get("points", [])]
    if not points:
        raise ValueError("profile points are empty")
    return points


def date_range_days(start_day: date, end_day: date) -> list[date]:
    days = pd.date_range(start_day, end_day, freq="D")
    return [ts.date() for ts in days]


def build_origin_days(point: BenchmarkPoint, pred_days: int) -> list[date]:
    origin_start = point.target_start - timedelta(days=pred_days)
    origin_end = point.target_end - timedelta(days=1)
    return date_range_days(origin_start, origin_end)


def build_train_groups(
    *,
    stage: str,
    point: BenchmarkPoint,
    pred_days: int,
    valid_days: int,
    rolling_interval_days: int,
    strategy_id: str,
    segmented: bool,
) -> list[TrainGroup]:
    origin_days = build_origin_days(point, pred_days=pred_days)
    if not origin_days:
        raise ValueError(f"point has no origin days: {point.point_id}")

    if stage in {"stage1", "stage3", "stage4"}:
        valid_origin_days = tuple(
            date_range_days(
                origin_days[0] - timedelta(days=valid_days),
                origin_days[0] - timedelta(days=1),
            )
        )
        return [
            TrainGroup(
                group_id="group_01",
                assigned_origin_days=tuple(origin_days),
                valid_origin_days=valid_origin_days,
                train_end_day=valid_origin_days[0] - timedelta(days=1),
                strategy_id=strategy_id,
                segmented=segmented,
            )
        ]

    groups: list[TrainGroup] = []
    group_index = 0
    for offset in range(0, len(origin_days), rolling_interval_days):
        assigned = tuple(origin_days[offset : offset + rolling_interval_days])
        if not assigned:
            continue
        group_index += 1
        first_origin = assigned[0]
        valid_origin_days = tuple(
            date_range_days(
                first_origin - timedelta(days=valid_days),
                first_origin - timedelta(days=1),
            )
        )
        groups.append(
            TrainGroup(
                group_id=f"group_{group_index:02d}",
                assigned_origin_days=assigned,
                valid_origin_days=valid_origin_days,
                train_end_day=valid_origin_days[0] - timedelta(days=1),
                strategy_id=strategy_id,
                segmented=segmented,
            )
        )
    return groups


def set_global_seed(seed: int) -> dict:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    info = {
        "seed": int(seed),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": False,
        "cuda_seeded": False,
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": None,
        "tf32_disabled": False,
    }
    try:
        import torch

        torch.manual_seed(seed)
        info["torch_seeded"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            info["cuda_seeded"] = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
            info["torch_deterministic_algorithms"] = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            info["cudnn_deterministic"] = True
            info["cudnn_benchmark"] = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
                info["tf32_disabled"] = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
            info["tf32_disabled"] = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    except Exception as exc:  # pragma: no cover - torch may be absent in local dry-run
        info["torch_seed_error"] = str(exc)
    return info


def current_git_info(repo_root: Path = REPO_ROOT) -> dict:
    def _run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                text=True,
                capture_output=True,
            )
        except Exception:
            return None
        return completed.stdout.strip() or None

    return {
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _run("rev-parse", "HEAD"),
        "status_short": _run("status", "--short"),
    }


def visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def selected_gpu_ids(device: str | None) -> list[str]:
    if not device:
        return []
    value = str(device)
    if not value.startswith("cuda"):
        return []
    slot_text = value.split(":", 1)[1] if ":" in value else "0"
    try:
        slot_index = int(slot_text)
    except ValueError:
        return [slot_text]
    visible = visible_gpu_ids()
    if visible and 0 <= slot_index < len(visible):
        return [visible[slot_index]]
    return [str(slot_index)]


def gpu_runtime_plan(device: str | None, gpu_limit: int) -> str:
    if not device or not str(device).startswith("cuda"):
        return f"cpu_only limit={gpu_limit}"
    selected = selected_gpu_ids(device)
    visible = visible_gpu_ids()
    return (
        f"selected_gpu_ids={selected or ['unknown']} "
        f"visible_gpu_ids={visible or ['default']} "
        f"planned_occupancy=1/{gpu_limit}"
    )


def machine_benchmark_root(profile: dict, machine: str) -> Path | None:
    machine_root = str(profile.get("machine_roots", {}).get(machine, "")).strip()
    if not machine_root:
        return None
    return Path(machine_root) / PACKAGE_ROOT.name


def machine_result_dir(profile: dict, machine: str, result_dir: Path) -> Path | None:
    benchmark_root = machine_benchmark_root(profile, machine)
    if benchmark_root is None:
        return None
    try:
        relative = result_dir.relative_to(PACKAGE_ROOT)
    except ValueError:
        return benchmark_root / result_dir.name
    return benchmark_root / relative


def planned_machine(profile: dict, stage: str, model_family: str) -> str:
    return str(profile.get("machine_plan", {}).get(stage, {}).get(model_family, "dl4"))


def infer_current_machine(profile: dict) -> str | None:
    hostname = socket.gethostname().strip().lower()
    for machine in profile.get("machine_roots", {}).keys():
        key = str(machine).strip().lower()
        if not key:
            continue
        if hostname == key or hostname.startswith(f"{key}.") or hostname.startswith(key):
            return str(machine)

    current_root = str(PACKAGE_ROOT.resolve())
    for machine in profile.get("machine_roots", {}).keys():
        benchmark_root = machine_benchmark_root(profile, str(machine))
        if benchmark_root is None:
            continue
        benchmark_root_text = str(benchmark_root.resolve())
        if current_root.startswith(benchmark_root_text):
            return str(machine)
    return None


def format_day_span(days: Iterable[date]) -> str:
    values = list(days)
    if not values:
        return "none"
    return f"{values[0].isoformat()}..{values[-1].isoformat()} (n={len(values)})"


def dry_run_lines(
    *,
    profile: dict,
    data_config: dict,
    profile_path: Path,
    stage: str,
    model_family: str,
    points: list[BenchmarkPoint],
    train_groups: dict[str, list[TrainGroup]],
    result_dir: Path,
    archive_dir: Path,
    planned_machine_name: str,
    machine: str,
    device: str | None,
    gpu_limit: int,
    audit_context: dict | None = None,
    nn_backbone_id: str | None = None,
    default_backbone_id: str | None = None,
    native_model_contract: dict | None = None,
    path_context: dict | None = None,
) -> str:
    path_context = path_context or {}
    path_sources = path_context.get("path_sources", {})
    path_existence = path_context.get("path_existence", {})

    def _exists_label(key: str) -> str:
        status = path_existence.get(key, {})
        return "yes" if status.get("exists") else "no"

    machine_root = str(profile.get("machine_roots", {}).get(machine, ""))
    target_benchmark_root = machine_benchmark_root(profile, machine)
    target_result_dir = machine_result_dir(profile, machine, result_dir)
    planned_machine_root = str(profile.get("machine_roots", {}).get(planned_machine_name, ""))
    planned_benchmark_root = machine_benchmark_root(profile, planned_machine_name)
    planned_result_dir = machine_result_dir(profile, planned_machine_name, result_dir)
    segment_ranges = profile.get("stage_defaults", {}).get(stage, {}).get("segment_ranges")
    if not segment_ranges:
        segment_ranges = [[1, int(profile["contract"]["pred_days"])]]
    segment_labels = [f"D{int(start)}-D{int(end)}" for start, end in segment_ranges]
    lines = [
        "Shandong D1-D14 Target-Day Benchmark dry-run",
        f"profile_id: {profile['profile_id']}",
        f"display_name: {profile.get('display_name', profile['profile_id'])}",
        f"profile_status: {profile.get('profile_status', 'formal')}",
        f"source_profile_id: {profile.get('source_profile_id', profile['profile_id'])}",
        f"availability_note: {profile.get('availability_note', 'none')}",
        f"profile_path: {profile_path}",
        f"stage: {stage}",
        f"model_family: {model_family}",
        f"nn_backbone_id: {nn_backbone_id or 'n/a'}",
        f"default_backbone_id: {default_backbone_id or 'n/a'}",
        f"segment_ranges: {', '.join(segment_labels)}",
        f"load_path: {data_config['load_path']}",
        f"load_path_source: {path_sources.get('load_path', 'profile')} exists={_exists_label('load_path')}",
        f"weather_root: {data_config['weather_root']}",
        f"weather_root_source: {path_sources.get('weather_root', 'profile')} exists={_exists_label('weather_root')}",
        f"weather_source: {data_config['weather_source']}",
        f"weather_source_source: {path_sources.get('weather_source', 'profile')}",
        f"weather_source_root: {path_context.get('resolved_paths', {}).get('weather_source_root') or 'unspecified'} exists={_exists_label('weather_source_root')}",
        f"cache_root: {data_config.get('merged_cache_root')}",
        f"cache_root_source: {path_sources.get('cache_root', 'profile')} exists={_exists_label('cache_root')}",
        f"result_root: {path_context.get('resolved_paths', {}).get('result_root') or resolve_path(profile['output']['result_root'], PACKAGE_ROOT)}",
        f"result_root_source: {path_sources.get('result_root', 'profile')} exists={_exists_label('result_root')}",
        f"paths_config: {path_context.get('paths_config_path') or 'not loaded'} status={path_context.get('paths_config_status', 'unknown')}",
        f"paths_config_example: {path_context.get('paths_config_example', EXAMPLE_PATHS_CONFIG)}",
        f"path_priority: {path_context.get('priority', 'CLI > env > paths_config > profile')}",
        f"seed: {profile.get('reproducibility', {}).get('seed', 3407)}",
        f"planned_machine: {planned_machine_name}",
        f"planned_machine_root: {planned_machine_root or 'unspecified'}",
        f"planned_machine_benchmark_root: {planned_benchmark_root or 'unspecified'}",
        f"planned_machine_result_dir: {planned_result_dir or 'unspecified'}",
        f"machine: {machine}",
        f"machine_root: {machine_root or 'unspecified'}",
        f"machine_benchmark_root: {target_benchmark_root or 'unspecified'}",
        f"device: {device or 'cpu'}",
        f"gpu_limit_per_machine: {gpu_limit}",
        f"gpu_plan: {gpu_runtime_plan(device, gpu_limit)}",
        f"execution_result_dir: {result_dir}",
        f"machine_result_dir: {target_result_dir or 'unspecified'}",
        f"desired_archive_dir: {archive_dir}",
        f"archive_root_source: {path_sources.get('archive_root', 'profile')} exists={_exists_label('archive_root')}",
        "archive_note: remote runs may not have controller archive disks mounted; controller-side copy can be required",
        f"official_files: {', '.join(profile['contract'].get('official_files', []))}",
        f"issue_id: {(audit_context or {}).get('issue_id') or 'none'}",
        f"change_id: {(audit_context or {}).get('change_id') or 'none'}",
        f"line_id: {(audit_context or {}).get('line_id') or 'none'}",
        f"compare_to_run_id: {(audit_context or {}).get('compare_to_run_id') or 'none'}",
        f"baseline_run_id: {(audit_context or {}).get('baseline_run_id') or 'none'}",
        f"decision_status: {(audit_context or {}).get('decision_status') or 'none'}",
        f"record_md_relpath: {(audit_context or {}).get('record_md_relpath') or 'none'}",
        f"targeted_eval_set: {', '.join((audit_context or {}).get('targeted_eval_set', [])) or 'none'}",
        f"control_only: {bool((audit_context or {}).get('control_only', False))}",
        "points:",
    ]
    if native_model_contract:
        effective_pred_len_points = native_model_contract.get("pred_len")
        if stage == "stage3":
            segment_pred_lines = []
            for start, end in segment_ranges:
                segment_pred_lines.append(f"D{int(start)}-D{int(end)}={(int(end) - int(start) + 1) * 96}")
            effective_pred_len_points = ", ".join(segment_pred_lines)
        lines.extend(
            [
                f"backbone_type: {native_model_contract.get('backbone_type')}",
                f"source_native_config_template: {native_model_contract.get('source_native_config_template')}",
                f"effective_pred_len_points: {effective_pred_len_points}",
                f"effective_gap_steps: {native_model_contract.get('gap_steps')}",
                f"only_output_after_gap: {native_model_contract.get('only_output_after_gap')}",
                f"day2_weight: {native_model_contract.get('day2_weight')}",
                f"midday_penalty: {native_model_contract.get('midday_penalty')}",
                f"time_embedding_type: {native_model_contract.get('time_embedding_type')}",
                f"use_multi_point_weather: {native_model_contract.get('use_multi_point_weather')}",
                f"use_holiday_features: {native_model_contract.get('use_holiday_features')}",
            ]
        )
    pred_days = int(profile["contract"]["pred_days"])
    for point in points:
        origin_days = build_origin_days(point, pred_days=pred_days)
        lines.append(
            f"  - {point.point_id}: target={point.target_start}..{point.target_end} "
            f"origin={origin_days[0]}..{origin_days[-1]} calls={len(origin_days)}"
        )
        for group in train_groups[point.point_id]:
            lines.append(
                f"    * {group.group_id}: strategy={group.strategy_id} "
                f"assigned={format_day_span(group.assigned_origin_days)} "
                f"planned_valid={format_day_span(group.valid_origin_days)} "
                f"train_end_day={group.train_end_day.isoformat()}"
            )
    return "\n".join(lines)


def build_result_dir(
    *,
    profile: dict,
    stage: str,
    model_family: str,
    run_tag: str | None,
    smoke: bool,
) -> Path:
    root = resolve_path(profile["output"]["result_root"], PACKAGE_ROOT)
    if smoke:
        root = root.parent / f"{root.name}_smoke"
    timestamp = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    template = profile["output"].get("run_name_template", "{profile_id}_{stage}_{model_family}_{timestamp}")
    name = template.format(
        profile_id=profile["profile_id"],
        stage=stage,
        model_family=model_family,
        timestamp=timestamp,
    )
    return root / name


def selected_machine(profile: dict, stage: str, model_family: str, override: str | None) -> str:
    if override:
        return override
    inferred = infer_current_machine(profile)
    if inferred:
        return inferred
    return planned_machine(profile, stage, model_family)


def ensure_day_tuple(values: Iterable[date]) -> tuple[date, ...]:
    return tuple(pd.Timestamp(value).date() for value in values)
