#!/bin/bash
# Full test-slice run (only after TEST_UNLOCKED). Each step is its own short GPU job under the shared lock.
set -e
X=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline/Milestone2/AudioXAI
A=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/audioxai
test -e $X/TEST_UNLOCKED
cd $X
$X/scripts/gpujob.sh w2_test_scan_a "audioxai scan test 50, chunk 1" $A scan --slice test --sample 50 --limit 25
$X/scripts/gpujob.sh w2_test_scan_b "audioxai scan test 50, chunk 2" $A scan --slice test --sample 50
$X/scripts/gpujob.sh w2_test_m2 "M2 occlusion test n=20" $A m2 --slice test --n 20
$X/scripts/gpujob.sh w2_test_e2e "e2e segment replacement test n=10" $A e2e --slice test --n 10
CUDA_VISIBLE_DEVICES="" $A summarize --slice test > /dev/null
CUDA_VISIBLE_DEVICES="" $A report --slice test --out report.html
