#!/bin/bash
# Coordinator priority 3: whisper-base as a second ASR variant on the same 50-clip test sample (results/whisper-base).
set -e
X=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline/Milestone2/AudioXAI
A=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/audioxai
cd $X
export AUDIOXAI_ASR=openai/whisper-base AUDIOXAI_RESULTS=results/whisper-base
$X/scripts/gpujob.sh w2_test_base_a "whisper-base test 50, chunk 1" $A scan --slice test --sample 50 --limit 25
$X/scripts/gpujob.sh w2_test_base_b "whisper-base test 50, chunk 2" $A scan --slice test --sample 50
CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
CUDA_VISIBLE_DEVICES="" $A report --slice test --out results/whisper-base/report_whisper-base.html
