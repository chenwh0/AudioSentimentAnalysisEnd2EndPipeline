import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser("audioxai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explain", help="explain one clip")
    e.add_argument("--clip", required=True)
    e.add_argument("--asr", default="openai/whisper-tiny")
    e.add_argument("--sentiment", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    e.add_argument("--out", default=None)
    e.add_argument("--smooth-n", type=int, default=8)
    s = sub.add_parser("scan", help="scan a slice sample (resumable)")
    s.add_argument("--slice", choices=["dev", "test"], default="dev")
    s.add_argument("--sample", type=int, default=50)
    s.add_argument("--limit", type=int, default=None, help="max clips this invocation (keeps GPU jobs short)")
    s.add_argument("--faith-deletion", type=int, default=20)
    s.add_argument("--faith-random", type=int, default=10)
    s.add_argument("--out", default=None, help="(informational) results dir is results/ for test, results/dev for dev")
    sm = sub.add_parser("summarize")
    sm.add_argument("--slice", choices=["dev", "test"], default="test")
    r = sub.add_parser("report", help="write the self-contained HTML report")
    r.add_argument("--slice", choices=["dev", "test"], default="test")
    r.add_argument("--out", default="report.html")
    r.add_argument("--max-clips", type=int, default=50)
    a = ap.parse_args(argv)
    if a.cmd == "explain":
        from . import core, paths
        if a.asr != paths.ASR_ID or a.sentiment != paths.SENT_ID:
            raise SystemExit("only openai/whisper-tiny + cardiffnlp/twitter-roberta-base-sentiment-latest are wired tonight")
        from .metrics import clip_metrics
        meta = core.clip_meta(a.clip)
        od = a.out or (paths.X / ("results/npz" if meta["slice"] == "test" else "results/dev/npz"))
        res = core.explain_clip(a.clip, smooth_n=a.smooth_n, save_dir=od)
        print(json.dumps(clip_metrics(res), default=float, indent=1))
    elif a.cmd == "scan":
        from .scan import scan
        left = scan(a.slice, a.sample, a.limit, a.faith_deletion, a.faith_random)
        print("remaining clips:", left)
    elif a.cmd == "summarize":
        from .scan import summarize
        print(json.dumps(summarize(a.slice), indent=1, default=float)[:4000])
    elif a.cmd == "report":
        from .report import build
        build(a.slice, a.out, a.max_clips)


if __name__ == "__main__":
    main()
