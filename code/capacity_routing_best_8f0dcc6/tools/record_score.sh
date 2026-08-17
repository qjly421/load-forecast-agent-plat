#!/usr/bin/env bash
# record_score.sh — called after each optimization iteration.
#
# Adapted from AutoSOTA's record_score.sh for EvoSOTA.
#
# Usage:
#   bash /path/to/record_score.sh \
#       --repo    /absolute/path/to/repo \
#       --scores  /path/to/scores.jsonl \
#       --iter    <N> \
#       --idea-id <IDEA-XXX|baseline> \
#       --title   "<idea title>" \
#       --status  <success|failed> \
#       --primary <float> \
#       --metrics '<json object>' \
#       --notes   "<optional notes>" \
#       --is-best <true|false>
#
# What it does:
#   0. Verify config/local_paths.yaml weather_source still matches the campaign
#      anchor (gitignored file, so protected_paths cannot cover it)
#   1. SHA256-verify protected files (if protected_paths is configured)
#   2. git add -A && git commit (captures current working-tree state)
#   3. On success AND improvement: moves the _best tag to this commit
#   4. Appends one JSON line to scores.jsonl with the real hash
#
# Exit codes: 0 = OK, 1 = argument error, 9 = protected file / weather_source violation

set -euo pipefail

# ── parse args ────────────────────────────────────────────────────────────────
REPO_ROOT=""
SCORES=""
ITER=""
IDEA_ID=""
TITLE=""
STATUS=""
PRIMARY=""
METRICS="{}"
EVAL_SCOPE=""
NOTES=""
IS_BEST=""   # "true" / "false" / "" (auto-detect)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO_ROOT="$2"; shift 2 ;;
    --scores)   SCORES="$2";   shift 2 ;;
    --iter)     ITER="$2";     shift 2 ;;
    --idea-id)  IDEA_ID="$2";  shift 2 ;;
    --title)    TITLE="$2";    shift 2 ;;
    --status)   STATUS="$2";   shift 2 ;;
    --primary)  PRIMARY="$2";  shift 2 ;;
    --metrics)  METRICS="$2";  shift 2 ;;
    --eval-scope) EVAL_SCOPE="$2"; shift 2 ;;
    --notes)    NOTES="$2";    shift 2 ;;
    --is-best)  IS_BEST="$2";  shift 2 ;;
    *) echo "[record_score] Unknown arg: $1" >&2; exit 1 ;;
  esac
done

for var in REPO_ROOT SCORES ITER IDEA_ID TITLE STATUS PRIMARY; do
  if [[ -z "${!var}" ]]; then
    echo "[record_score] Missing required arg: --${var,,}" >&2
    exit 1
  fi
done

# ── derive run_dir from scores path (scores → results → run_dir) ────────────
RUN_DIR="$(cd "$(dirname "$(dirname "$SCORES")")" && pwd)"

# ── weather_source anchor guard (cross-island comparability) ───────────────
# config/local_paths.yaml is gitignored, so it can never be carried by
# protected_paths (whose rollback story is "git checkout -- <file>"). But scores
# are only comparable across islands if every island evaluates on the same
# weather lineage, so the value is checked mechanically here on every record,
# baseline included. Override the anchor with EVOSOTA_WEATHER_SOURCE_EXPECTED;
# set that variable to the empty string to disable the guard.
WS_EXPECTED="${EVOSOTA_WEATHER_SOURCE_EXPECTED-concat_all3}"
WS_FILE="$REPO_ROOT/config/local_paths.yaml"

if [[ -n "$WS_EXPECTED" && -f "$WS_FILE" ]]; then
  set +e
  WS_ACTUAL="$(python3 -c 'import sys,yaml;print(str((yaml.safe_load(open(sys.argv[1])) or {}).get("weather_source","<missing>")).strip())' "$WS_FILE" 2>/dev/null)"
  if [[ -z "$WS_ACTUAL" ]]; then
    WS_ACTUAL="$(sed -n 's/^[[:space:]]*weather_source:[[:space:]]*\([^#[:space:]]*\).*/\1/p' "$WS_FILE" | tail -1)"
  fi
  set -e
  if [[ "$WS_ACTUAL" != "$WS_EXPECTED" ]]; then
    echo "[record_score] PROTOCOL VIOLATION -- weather_source drifted from the campaign anchor:"
    echo "    file    : $WS_FILE"
    echo "    expected: $WS_EXPECTED"
    echo "    actual  : ${WS_ACTUAL:-<unreadable>}"
    echo "[record_score] All islands must evaluate on one weather lineage; a score produced"
    echo "[record_score] under a different weather_source is not comparable to any other."
    echo "[record_score] This iteration is REJECTED. Restore the anchor:"
    echo "    sed -i 's/^weather_source:.*/weather_source: ${WS_EXPECTED}/' \"$WS_FILE\""
    exit 9
  fi
  echo "[record_score] weather_source OK ($WS_ACTUAL)"
fi

# ── protected-paths SHA256 verification ──────────────────────────────────────
# Reads protected_paths from config.yaml in the output directory.
# Snapshots SHA256 on iter=0; on every later iteration verifies hashes match.
# Mismatch → exit 9 (PROTOCOL VIOLATION).
PROTECTED_HASH_FILE="$REPO_ROOT/.evosota_protected_hashes.json"
OUTPUT_DIR="$(cd "$(dirname "$(dirname "$SCORES")")" && pwd)"
EFFECTIVE_CFG="$OUTPUT_DIR/config.yaml"

if [[ -f "$EFFECTIVE_CFG" ]]; then
  set +e
  python3 - "$EFFECTIVE_CFG" "$REPO_ROOT" "$PROTECTED_HASH_FILE" "$ITER" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

import yaml

cfg_path, repo_str, hash_file, iter_str = sys.argv[1:5]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

protected = cfg.get("protected_paths") or []
if not protected:
    sys.exit(0)

repo = Path(repo_str)


def compute_hashes(rel_paths):
    out = {}
    for rel in rel_paths:
        full = (repo / rel).resolve()
        if full.is_file():
            out[rel] = hashlib.sha256(full.read_bytes()).hexdigest()
        elif full.is_dir():
            for sub in sorted(full.rglob("*")):
                if sub.is_file():
                    key = str(sub.relative_to(repo))
                    out[key] = hashlib.sha256(sub.read_bytes()).hexdigest()
        else:
            out[rel] = "<missing>"
    return out


cur = compute_hashes(protected)
hash_path = Path(hash_file)
is_baseline = iter_str in ("0", "baseline")

if is_baseline or not hash_path.exists():
    hash_path.write_text(json.dumps(cur, indent=2, sort_keys=True))
    print(f"[record_score] protected snapshot stored: {len(cur)} file(s) -> {hash_path}")
    sys.exit(0)

baseline = json.loads(hash_path.read_text())
diffs = sorted(k for k in set(cur) | set(baseline) if cur.get(k) != baseline.get(k))

if not diffs:
    print(f"[record_score] protected files OK ({len(cur)} checked)")
    sys.exit(0)

print("[record_score] PROTOCOL VIOLATION -- protected file(s) modified:")
for k in diffs:
    was = (baseline.get(k) or "<absent>")[:12]
    now = (cur.get(k) or "<deleted>")[:12]
    print(f"    - {k}   ({was} -> {now})")
print("[record_score] This iteration is REJECTED. Roll back protected files:")
print("    cd <repo> && git checkout -- <file>")
sys.exit(9)
PYEOF
  PROTECT_RC=$?
  set -e
  if [[ "$PROTECT_RC" -ne 0 ]]; then
    exit "$PROTECT_RC"
  fi
fi

# ── idempotency: skip if this iter+idea already recorded ─────────────────────
ALREADY_RECORDED=false
if [[ -f "$SCORES" ]] && [[ "$ITER" != "final" ]]; then
  if python3 - "$SCORES" "$ITER" "$IDEA_ID" <<'PYEOF' >/dev/null 2>&1
import json, sys
scores, want_iter, want_idea = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    want_iter_norm = int(want_iter)
except ValueError:
    want_iter_norm = want_iter
for line in open(scores):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get("iter") == want_iter_norm and r.get("idea_id") == want_idea:
        sys.exit(0)
sys.exit(1)
PYEOF
  then
    ALREADY_RECORDED=true
    echo "[record_score] iter=${ITER} idea=${IDEA_ID} already in scores.jsonl -- skipping commit/append."
  fi
fi

# ── git commit (capture current state) ───────────────────────────────────────
cd "$REPO_ROOT"

git config user.name  "optimizer" 2>/dev/null || true
git config user.email "opt@local" 2>/dev/null || true

if [[ "$ALREADY_RECORDED" == "false" ]]; then
  git add -A
  COMMIT_MSG="iter-${ITER}: ${TITLE} [${STATUS}]"
  git commit -q -m "$COMMIT_MSG" --allow-empty
  COMMIT_HASH=$(git rev-parse HEAD)
  echo "[record_score] git commit: ${COMMIT_HASH:0:10} -- $COMMIT_MSG"
else
  COMMIT_HASH=$(git rev-parse HEAD)
fi

# ── update _best tag ─────────────────────────────────────────────────────────
if [[ "$STATUS" == "success" ]]; then
  TAG_EXISTS=$(git rev-parse --verify _best >/dev/null 2>&1 && echo yes || echo no)

  if [[ "$TAG_EXISTS" == "no" ]]; then
    git tag -f _best "$COMMIT_HASH"
    echo "[record_score] _best tag created -> ${COMMIT_HASH:0:10} (first success)"

  elif [[ "$IS_BEST" == "true" ]]; then
    git tag -f _best "$COMMIT_HASH"
    echo "[record_score] _best tag updated -> ${COMMIT_HASH:0:10}"

  elif [[ -z "$IS_BEST" ]]; then
    echo "[record_score] WARNING: --is-best not provided; _best tag unchanged"
  fi
fi

# ── append to scores.jsonl ────────────────────────────────────────────────────
mkdir -p "$(dirname "$SCORES")"

NOTES_ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$NOTES" 2>/dev/null \
  || echo "\"${NOTES//\"/\'}\"")
TITLE_ESCAPED=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$TITLE" 2>/dev/null \
  || echo "\"${TITLE//\"/\'}\"")

if [[ "$ALREADY_RECORDED" == "false" ]]; then
  METRICS_JSON="$METRICS" python3 - <<PYEOF
import json, os
raw_iter = "$ITER"
try:
    iter_val = int(raw_iter)
except ValueError:
    iter_val = raw_iter

# ── Canonical metrics schema (Q1): normalize the agent-supplied metrics so every
#    scores.jsonl row has the SAME shape regardless of which key the agent used.
#    Per-horizon overalls are extracted into FLAT D{k}_overall (k=1..5) from any
#    of: flat D{k}_overall, bare D{k} scalar, nested D{k}:{overall}, or
#    horizon_detail.D{k}.overall.
#    The BD code (bd.py) then reads ONLY the flat canonical keys — no endless
#    adaptation to varied storage styles.
def _horizon_overall(m, label):
    v = m.get(f"{label}_overall")
    if isinstance(v, (int, float)):
        return float(v)
    d = m.get(label)
    if isinstance(d, dict):
        v = d.get("overall", d.get("overall_acc"))
        if isinstance(v, (int, float)):
            return float(v)
    elif isinstance(d, (int, float)):
        # bare D{k} scalar (e.g. {"D1": 96.96}) = the horizon overall directly.
        # Some agents (notably resuming carry islands anchored to a legacy
        # recording style) emit per-horizon as bare scalars; normalize here so
        # BD never sees an `unknown` horizon_degradation from this form.
        return float(d)
    hd = m.get("horizon_detail")
    if isinstance(hd, dict):
        d2 = hd.get(label) or hd.get(label.replace("D", ""))
        if isinstance(d2, dict):
            v = d2.get("overall", d2.get("overall_acc"))
            if isinstance(v, (int, float)):
                return float(v)
    return None

try:
    metrics = json.loads(os.environ.get("METRICS_JSON") or "{}")
except Exception:
    metrics = {}
metrics = dict(metrics)  # preserve all original keys
for _k in ("D1", "D2", "D3", "D4", "D5"):
    _val = _horizon_overall(metrics, _k)
    if _val is not None:
        metrics[f"{_k}_overall"] = _val

# ── eval_scope (Q5): tag every row medium|full so BD/portfolio can filter to the
#    per-iter medium eval (the optimization target) and full-Δ reporting is
#    reliable. Source priority: (1) the agent's --eval-scope arg (it ran the eval,
#    knows the mode); (2) metrics.eval_scope (if the agent propagated it from the
#    eval_wrapper line); (3) INFER from midday (medium midday ~94, full ~91 on
#    this task — task-specific fallback that catches agent non-compliance, e.g. a
#    `final` row whose metrics are actually medium); (4) "unknown".
_arg_scope = ("$EVAL_SCOPE" or "").strip().lower()
_mid = metrics.get("midday")
try:
    _mid = float(_mid)
except (TypeError, ValueError):
    _mid = None
if _arg_scope in ("medium", "full"):
    eval_scope = _arg_scope
elif isinstance(metrics.get("eval_scope"), str) and metrics["eval_scope"].lower() in ("medium", "full"):
    eval_scope = metrics["eval_scope"].lower()
elif _mid is not None:
    eval_scope = "medium" if _mid >= 92.5 else "full"   # task-specific inference fallback
else:
    eval_scope = "unknown"

entry = {
    "iter":           iter_val,
    "idea_id":        "$IDEA_ID",
    "idea_title":     "$TITLE",
    "metrics":        metrics,
    "primary_metric": float("$PRIMARY"),
    "commit":         "$COMMIT_HASH",
    "status":         "$STATUS",
    "eval_scope":     eval_scope,
    "notes":          "$NOTES",
}
scores_path = "$SCORES"
with open(scores_path, "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"[record_score] Appended iter={entry['iter']} to {scores_path}")
PYEOF
fi

echo "[record_score] Done."
