#!/bin/bash
R=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline
PY=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python
CLIPS="id10287_2iaCNe7ajII_00004 id10285_uArtiTSTnSU_00014 id10296_f_k09R8r_cA_00004 id10288_R4S2c-NiPfA_00001 id10273_ZpPPaMazpf0_00003"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $R/Milestone2/AudioXAI
echo "{\"ts\":\"$(date -Is)\",\"job\":\"xai_tok_m2\",\"event\":\"start\",\"pid\":$$,\"note\":\"M2 occlusion, whisper-base + AventIQ, 5 representative clips\"}" >> $R/logs/gpu_jobs.jsonl
flock /tmp/m2_gpu.lock bash -c "
AUDIOXAI_ASR=openai/whisper-base $PY scripts/m2_for_clips.py whisper-base $CLIPS
AUDIOXAI_ASR=AventIQ-AI/whisper-audio-to-text $PY scripts/m2_for_clips.py aventiq $CLIPS
"
rc=$?
echo "{\"ts\":\"$(date -Is)\",\"job\":\"xai_tok_m2\",\"event\":\"end\",\"pid\":$$,\"note\":\"rc=$rc\"}" >> $R/logs/gpu_jobs.jsonl
echo "$(date -Is) | XAI-TOK | GPU job xai_tok_m2 finished rc=$rc" >> $R/logs/ledger.md
