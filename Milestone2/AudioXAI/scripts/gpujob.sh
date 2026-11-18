#!/bin/bash
# usage: gpujob.sh <jobname> <note> cmd...
# Runs cmd under the shared GPU flock and appends start/end JSON lines to logs/gpu_jobs.jsonl
R=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline
J="$1"; N="$2"; shift 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
flock -w "${FLOCK_WAIT:-86400}" /tmp/m2_gpu.lock bash -c '
J="$1"; N="$2"; L="$3"; shift 3
echo "{\"ts\": \"$(date +%Y-%m-%dT%H:%M:%S%z)\", \"job\": \"$J\", \"event\": \"start\", \"pid\": $$, \"note\": \"$N\"}" >> "$L"
"$@"; rc=$?
echo "{\"ts\": \"$(date +%Y-%m-%dT%H:%M:%S%z)\", \"job\": \"$J\", \"event\": \"end\", \"pid\": $$, \"note\": \"$N rc=$rc\"}" >> "$L"
exit $rc' _ "$J" "$N" "$R/logs/gpu_jobs.jsonl" "$@"
