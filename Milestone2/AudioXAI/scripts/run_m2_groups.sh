#!/bin/bash
R=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline
PY=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $R/Milestone2/AudioXAI
echo "{\"ts\":\"$(date -Is)\",\"job\":\"xai_tok_m2_groups\",\"event\":\"start\",\"pid\":$$,\"note\":\"M2 occlusion, whisper-base + AventIQ, 20-clip study set\"}" >> $R/logs/gpu_jobs.jsonl
flock /tmp/m2_gpu.lock bash -c "
AUDIOXAI_ASR=openai/whisper-base $PY scripts/m2_for_clips.py whisper-base id10273_Ed918auNI0Y_00004 id10273_E50E5hbggyM_00005 id10288_FIxC_PNJNkc_00003 id10298_6qFnVechX9o_00012 id10287_2iaCNe7ajII_00002 id10273_ZpPPaMazpf0_00008 id10291_iPZGPcrYIZs_00007 id10276_HrA2BLcLApA_00026 id10300_8Sz2-IYJ2GA_00013 id10271_zWyD72sHVwA_00007 id10273_hW0Q6eHqN-U_00004 id10281_1CiBWgsHT4E_00001 id10285_uArtiTSTnSU_00014 id10309_XMLMvfrgdzY_00009 id10276_3vWez3baO60_00009 id10273_8fKRoUIpAJY_00012 id10287_V0ubcXO71Sg_00001 id10304_W-5dGkFnQxM_00001 id10287_2iaCNe7ajII_00004 id10284_Uzxv7Axh3Z8_00006
AUDIOXAI_ASR=AventIQ-AI/whisper-audio-to-text $PY scripts/m2_for_clips.py aventiq id10273_Ed918auNI0Y_00004 id10273_E50E5hbggyM_00005 id10288_FIxC_PNJNkc_00003 id10298_6qFnVechX9o_00012 id10287_2iaCNe7ajII_00002 id10273_ZpPPaMazpf0_00008 id10291_iPZGPcrYIZs_00007 id10276_HrA2BLcLApA_00026 id10300_8Sz2-IYJ2GA_00013 id10271_zWyD72sHVwA_00007 id10273_hW0Q6eHqN-U_00004 id10281_1CiBWgsHT4E_00001 id10285_uArtiTSTnSU_00014 id10309_XMLMvfrgdzY_00009 id10276_3vWez3baO60_00009 id10273_8fKRoUIpAJY_00012 id10287_V0ubcXO71Sg_00001 id10304_W-5dGkFnQxM_00001 id10287_2iaCNe7ajII_00004 id10284_Uzxv7Axh3Z8_00006
"
rc=$?
echo "{\"ts\":\"$(date -Is)\",\"job\":\"xai_tok_m2_groups\",\"event\":\"end\",\"pid\":$$,\"note\":\"rc=$rc\"}" >> $R/logs/gpu_jobs.jsonl
echo "$(date -Is) | XAI-TOK | GPU job xai_tok_m2_groups finished rc=$rc" >> $R/logs/ledger.md
