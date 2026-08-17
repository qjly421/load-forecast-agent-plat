#!/usr/bin/env bash
# fingerprint_check.sh — Verify protected file SHA256 checksums.
#
# Standalone utility to check whether any protected files have been modified
# since the baseline snapshot. Used as a quick sanity check before evaluation.
#
# Usage:
#   bash fingerprint_check.sh \
#       --repo /path/to/repo \
#       --hashes /path/to/.evosota_protected_hashes.json \
#       --config /path/to/config.yaml
#
# Exit codes: 0 = all OK, 1 = error, 9 = violation detected

set -euo pipefail

REPO_ROOT=""
HASH_FILE=""
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)   REPO_ROOT="$2"; shift 2 ;;
    --hashes) HASH_FILE="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    *) echo "[fingerprint_check] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$REPO_ROOT" ]] || [[ -z "$HASH_FILE" ]]; then
  echo "[fingerprint_check] Missing --repo or --hashes" >&2
  exit 1
fi

if [[ ! -f "$HASH_FILE" ]]; then
  echo "[fingerprint_check] No baseline hash file found: $HASH_FILE"
  echo "[fingerprint_check] Run a baseline iteration first to create snapshots."
  exit 0
fi

# Run the Python verification
python3 - "$REPO_ROOT" "$HASH_FILE" "$CONFIG_FILE" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

import yaml

repo_str, hash_file, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]

# Load baseline hashes
baseline = json.loads(Path(hash_file).read_text())
repo = Path(repo_str)

# Optionally load protected_paths from config for re-verification
protected = []
if cfg_path and Path(cfg_path).is_file():
    cfg = yaml.safe_load(Path(cfg_path).read_text()) or {}
    protected = cfg.get("protected_paths") or []

if not protected and baseline:
    # If no config but hash file exists, check all keys in baseline
    protected = list(baseline.keys())

def compute_hash(p):
    full = (repo / p).resolve()
    if full.is_file():
        return hashlib.sha256(full.read_bytes()).hexdigest()
    elif full.is_dir():
        # For directories, return hash of concatenated file hashes
        h = hashlib.sha256()
        for sub in sorted(full.rglob("*")):
            if sub.is_file():
                h.update(sub.read_bytes())
        return h.hexdigest()
    return "<missing>"

violations = []
for rel_path, expected_hash in baseline.items():
    current = compute_hash(rel_path)
    if current != expected_hash:
        violations.append((rel_path, expected_hash[:12], current[:12]))

if violations:
    print("[fingerprint_check] VIOLATION DETECTED:")
    for path, was, now in violations:
        print(f"  - {path}  ({was} -> {now})")
    sys.exit(9)
else:
    print(f"[fingerprint_check] All {len(baseline)} protected file(s) verified OK")
    sys.exit(0)
PYEOF
