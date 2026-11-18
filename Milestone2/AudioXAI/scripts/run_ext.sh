#!/bin/bash
# Extensions after the Phase 4 commit; writes only results/n300, results/aventiq, results/n100 (M2).
set -e
X=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline/Milestone2/AudioXAI
A=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/audioxai
cd $X
NOTE300="n=300 stratified test sample by the same seeded procedure (sort by original_index, proportional allocation, labels alphabetical, default_rng(8001).permutation within label); the n=50 and n=100 samples are per-label prefixes of these permutations, hence subsets by construction. Rows of the 100 reused."
for i in 1 2 3 4; do
  AUDIOXAI_RESULTS=results/n300 $X/scripts/gpujob.sh w2_n300_$i "audioxai scan test n=300 chunk $i (<=55 clips)" $A scan --slice test --sample 300 --limit 55
done
AUDIOXAI_RESULTS=results/n300 $X/scripts/gpujob.sh w2_n300_5 "audioxai scan test n=300 final chunk" $A scan --slice test --sample 300
AUDIOXAI_RESULTS=results/n300 AUDIOXAI_NOTE="$NOTE300" CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
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
