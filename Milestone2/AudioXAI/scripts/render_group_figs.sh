#!/bin/bash
# usage: render_group_figs.sh <model>   (renders tokens + summary figures for the 20 study clips, CPU only)
R=/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline
PY=/home/cppesl/Desktop/Alve/e2eML_course/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
cd $R
python3 -c "import pandas as pd; d=pd.read_csv('Milestone2/AudioXAI/results/token_segments/groups.csv'); d=d[d.group!='background']; [print(r['clip'], r.degradation_class, r.group) for _,r in d.iterrows()]" | while read c cls grp; do
  $PY figures/render_xai_tokens.py --model $1 --clip $c --cls "$cls ($grp set)" 2>&1 | grep -v -i warn | tail -1
done
echo "$(date -Is) | XAI-TOK | figures rendered for $1 on the 20 study clips" >> $R/logs/ledger.md
