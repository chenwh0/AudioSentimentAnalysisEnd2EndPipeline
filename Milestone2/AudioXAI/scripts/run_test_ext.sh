#!/bin/bash
# Coordinator priority 2: extend to 100 clips (results/n100; the 50-clip sample is a per-label prefix subset by
# construction, rows for those 50 are reused) and M2 occlusion on all 50 clips of the primary sample.
set -e
X=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline/Milestone2/AudioXAI
A=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/audioxai
cd $X
$X/scripts/gpujob.sh w2_test_m2_50 "M2 occlusion on all 50 test-sample clips" $A m2 --slice test --n 50
AUDIOXAI_RESULTS=results/n100 $X/scripts/gpujob.sh w2_test100_scan "audioxai scan test 100 (50 reused)" $A scan --slice test --sample 100
CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
CUDA_VISIBLE_DEVICES="" $A report --slice test --out report.html
AUDIOXAI_RESULTS=results/n100 CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
