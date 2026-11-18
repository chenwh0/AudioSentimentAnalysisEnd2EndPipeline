import os
from pathlib import Path

R = Path(os.environ.get("AUDIOXAI_REPO", "/home/cppesl/Desktop/Alve/e2eML_course/AudioSentimentAnalysisEnd2EndPipeline"))
PE = R / "Milestone2/PipelineEvaluation"
AUDIO = PE / "data/audio"
GOLD_TSV = R / "Milestone2/T1_final/data/slue_voxceleb_test.tsv"
X = R / "Milestone2/AudioXAI"
T1RES = R / "Milestone2/T1_final/results"
AADD_READY = T1RES / "AADD_READY"
AADD_SEG = T1RES / "aadd_devtest_seg20ms.csv"
TEST_UNLOCKED = X / "TEST_UNLOCKED"
WSP_READY = R / "Milestone2/WSP_Whisper/splits/READY"
WSP_HOLDOUT = R / "Milestone2/WSP_Whisper/splits/wsp_holdout_ids.npy"

ASR_ID = "openai/whisper-tiny"
ASR_REV = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
SENT_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENT_REV = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
SMOKE = ["id10281_RDFdX3VxjUQ_00021", "id10291_oLJhH76d0kU_00001"]
