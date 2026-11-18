#!/bin/bash
# usage: gpu_run.sh <job-name> <note> <script.py> [args...]  -- flock + start/end lines in logs/gpu_jobs.jsonl
R=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline
P=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python
job=$1; note=$2; shift 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
flock /tmp/m2_gpu.lock bash -c '
  R='"$R"'; job="$1"; note="$2"; shift 2
  echo "{\"ts\":\"$(date -Iseconds)\",\"job\":\"$job\",\"event\":\"start\",\"pid\":$$,\"note\":\"$note\"}" >> $R/logs/gpu_jobs.jsonl
  '"$P"' "$@"; rc=$?
  echo "{\"ts\":\"$(date -Iseconds)\",\"job\":\"$job\",\"event\":\"end\",\"pid\":$$,\"note\":\"$note rc=$rc\"}" >> $R/logs/gpu_jobs.jsonl
  exit $rc' _ "$job" "$note" "$@"
