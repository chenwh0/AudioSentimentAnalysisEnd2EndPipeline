#!/bin/bash
set -e
X=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline/Milestone2/AudioXAI
A=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/audioxai
cd $X
export AUDIOXAI_ASR=AventIQ-AI/whisper-audio-to-text
AUDIOXAI_RESULTS=results/aventiq $X/scripts/gpujob.sh w2_aventiq_a "AventIQ test 50 chunk 1" $A scan --slice test --sample 50 --limit 25
AUDIOXAI_RESULTS=results/aventiq $X/scripts/gpujob.sh w2_aventiq_b "AventIQ test 50 chunk 2" $A scan --slice test --sample 50
AUDIOXAI_RESULTS=results/aventiq AUDIOXAI_NOTE="AventIQ-AI/whisper-audio-to-text (whisper-base fine-tune; alignment heads inherited from its generation_config) on the same 50 clips" CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
AUDIOXAI_RESULTS=results/aventiq CUDA_VISIBLE_DEVICES="" $A report --slice test --out results/aventiq/report_aventiq.html
unset AUDIOXAI_ASR
for i in 1 2 3 4; do
  AUDIOXAI_RESULTS=results/n100 $X/scripts/gpujob.sh w2_n100_m2_$i "M2 occlusion on n100 chunk $i" $A m2 --slice test --n $((25*i))
done
AUDIOXAI_RESULTS=results/n100 AUDIOXAI_NOTE="n=100 sample; the n=50 sample is a per-label prefix subset by construction" CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
