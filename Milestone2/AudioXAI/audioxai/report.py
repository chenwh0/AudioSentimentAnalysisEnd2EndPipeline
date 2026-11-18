"""Self-contained offline HTML report (inline JS + canvas/SVG, no CDN, no audio)."""
import base64
import json

import numpy as np
import pandas as pd

from . import paths
from .core import CLASSES


def _q8(a):
    a = np.asarray(a, np.float32)
    lo, hi = float(np.nanmin(a)) if a.size else 0.0, float(np.nanmax(a)) if a.size else 1.0
    q = np.zeros(a.shape, np.uint8) if hi <= lo else np.round((a - lo) / (hi - lo) * 255).astype(np.uint8)
    return dict(b=base64.b64encode(q.tobytes()).decode(), shape=list(a.shape), lo=lo, hi=hi)


def _pool_time(a, cols):
    """pool last axis to at most `cols` columns by mean."""
    T = a.shape[-1]
    if T <= cols:
        return a
    edges = np.linspace(0, T, cols + 1).astype(int)
    return np.stack([a[..., edges[i]:max(edges[i + 1], edges[i] + 1)].mean(-1) for i in range(cols)], -1)


def clip_payload(npz_path, wav_env=None, cols=320):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    mel = z["logmel"].astype(np.float32)
    nf = mel.shape[1]
    c = min(cols, nf)
    m1 = np.abs(z["m1"].astype(np.float32))  # (K,80,nf)
    K = m1.shape[0]
    m1b = m1.reshape(K, 10, 8, nf).sum(2) if K else np.zeros((0, 10, nf))  # 10 bands of 8 mel bins
    m1b = _pool_time(m1b, c)
    if K:
        m1b = m1b / np.maximum(m1b.reshape(K, -1).max(1)[:, None, None], 1e-12)
        m1mean = m1b.mean(0, keepdims=True); m1mean /= max(m1mean.max(), 1e-12)
        m1b = np.concatenate([m1mean, m1b])
    m4 = z["m4"].astype(np.float32)
    m4 = _pool_time(np.repeat(m4, 2, -1)[:, :nf], c) if K else np.zeros((0, c))
    if K:
        m4 = m4 / np.maximum(m4.max(1, keepdims=True), 1e-12)
        mm = m4.mean(0, keepdims=True); m4 = np.concatenate([mm / max(mm.max(), 1e-12), m4])
    m3 = z["m3"].astype(np.float32) if "m3" in z else np.zeros((0, m4.shape[-1]))
    m3 = _pool_time(np.repeat(m3, 2, -1)[:, :nf], c) if len(m3) else np.zeros((0, c))
    if len(m3):
        m3 = m3 / np.maximum(m3.max(1, keepdims=True), 1e-12)
        mm = m3.mean(0, keepdims=True); m3 = np.concatenate([mm / max(mm.max(), 1e-12), m3])
    aadd = z["aadd"].astype(np.float32)
    aadd_p = _pool_time(np.repeat(aadd.T, 2, -1)[:, :nf], c) if aadd.size else np.zeros((12, c))
    flag = _pool_time(np.repeat((aadd >= 0.5).any(1).astype(np.float32), 2)[:nf], c) if aadd.size else np.zeros(c)
    import soundfile as sf
    wav, _ = sf.read(paths.AUDIO / f"{meta['meta']['file']}.wav", dtype="float32")
    env_cols = 600
    e = np.linspace(0, len(wav), env_cols + 1).astype(int)
    env = np.array([[wav[e[i]:e[i + 1]].min(), wav[e[i]:e[i + 1]].max()] for i in range(env_cols)])
    from .core import LABELS3
    seq = z["seq"]; keep = z["keep"]
    from .core import asr
    tok = asr()[0].tokenizer
    toks = [tok.decode([int(seq[t + 1])]) for t in keep]
    ab = z["ig_abs"].astype(float); sg = z["ig_sig"].astype(float)
    words = [dict(w=w["word"], t0=round(w["t0"], 2), t1=round(w["t1"], 2), tag=tg,
                  imp=round(float(ab[i] / ab.max()), 3) if len(ab) and ab.max() > 0 else 0,
                  sgn=1 if (len(sg) and sg[i] >= 0) else -1)
             for i, (w, tg) in enumerate(zip(meta["words"], meta["tags"]))]
    return dict(file=meta["meta"]["file"], label=meta["meta"]["label"], dur=meta["dur"], text=meta["text"],
                gold=meta["gold"], cnt=meta["cnt"], pred=LABELS3[int(np.argmax(z["probs"]))],
                probs=[round(float(x), 3) for x in z["probs"]],
                pred_gold=LABELS3[int(np.argmax(z["probs_gold"]))], toks=toks, words=words,
                mel=_q8(_pool_time(mel, c)), m1=_q8(m1b), m4=_q8(m4), m3=_q8(m3), aadd=_q8(aadd_p), flag=[round(float(v), 2) for v in flag],
                env=[[round(float(a), 3), round(float(b), 3)] for a, b in env], cols=c)


def build(slice_="test", out="report.html", max_clips=50):
    from .scan import out_dir
    od = out_dir(slice_)
    S = json.load(open(od / "summary.json"))
    agg = pd.read_csv(od / "aggregate.csv")
    faith = pd.read_csv(od / "faithfulness.csv") if (od / "faithfulness.csv").exists() else pd.DataFrame()
    clips = []
    for f in list(agg.file)[:max_clips]:
        p = od / "npz" / f"{f}.npz"
        if p.exists():
            clips.append(clip_payload(p))
    cols = [c for c in ["file", "label", "pred", "pred_gold_text", "duration", "wer", "frac_flagged", "degradation_class",
                        "focus_ratio_m1", "focus_ratio_m4", "wer_flagged", "wer_clean", "sent_imp_share_flagged",
                        "err_word_share", "err_word_imp_share"] if c in agg]
    table = json.loads(agg[cols].round(3).to_json(orient="records"))
    data = dict(summary=S, table=table, clips=clips, classes=CLASSES,
                faith=json.loads(faith.round(4).to_json(orient="records")) if len(faith) else [])
    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=float).replace("</", "<\\/"))
    outp = paths.X / out if not str(out).startswith("/") else out
    open(outp, "w").write(html)
    print("wrote", outp, len(html) // 1024, "KB,", len(clips), "clips")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AudioXAI Report</title>
<style>
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deded8;--acc:#2f6fb0;--bad:#c0392b;--ok:#2e8b57;--ins:#8e44ad;--sub:#d35400;--del:#7f8c8d}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--fg:#ececea;--mut:#a3a39d;--card:#1f1f1e;--line:#3a3a37;--acc:#6aa5e0;--bad:#e57366;--ok:#5cc08a;--ins:#b98ad4;--sub:#f0a060;--del:#a0a8aa}}
:root[data-theme="dark"]{--bg:#161615;--fg:#ececea;--mut:#a3a39d;--card:#1f1f1e;--line:#3a3a37;--acc:#6aa5e0;--bad:#e57366;--ok:#5cc08a;--ins:#b98ad4;--sub:#f0a060;--del:#a0a8aa}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1150px;margin:auto}
h1{font-size:1.5em;margin:.2em 0}h2{font-size:1.15em;margin:1.2em 0 .4em}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0;overflow-x:auto}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;white-space:nowrap}
th{cursor:pointer;color:var(--mut)}.mut{color:var(--mut);font-size:13px}.v{font-weight:600}
canvas{width:100%;display:block;image-rendering:pixelated;border:1px solid var(--line)}
select,button{font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:2px 6px}
.w{padding:1px 2px;margin:1px;border-radius:3px;display:inline-block}.tag-sub{outline:2px solid var(--sub)}.tag-ins{outline:2px dashed var(--ins)}
.lg span{margin-right:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
</style></head><body>
<h1 id="ttl">AudioXAI</h1>
<div class="mut" id="sub"></div>
<div class="card" id="answer"></div>
<h2>Hypotheses and verdicts</h2><div class="grid" id="verd"></div>
<h2>Faithfulness</h2><div class="card" id="faith"></div>
<div class="card"><svg id="delc" width="100%" height="190"></svg><div class="mut">Deletion curves: mean drop in per-token teacher-forced log-prob (nats) after removing the top-k% M1 patches (solid) vs the same number of random patches (dashed).</div></div>
<h2>Focus ratio per clip (M1)</h2><div class="card"><svg id="hist" width="100%" height="170"></svg><div class="mut">Focus ratio = M1 attribution mass in AADD-flagged 20 ms frames ÷ fraction of frames flagged; 1 = no concentration. Clips with no flagged frame are NaN and not plotted.</div></div>
<h2>Clip view</h2>
<div class="card">
<div>Clip <select id="clipSel"></select> Method <select id="meth"><option value="m1">M1 SmoothGrad grad×input (time×freq)</option><option value="m3">M3 Grad-CAM, last encoder layer (time-only)</option><option value="m4">M4 cross-attention, alignment heads (time-only)</option><option value="none">none</option></select>
Token <select id="tokSel"></select></div>
<div id="clipHead" style="margin:6px 0"></div>
<div class="mut">Waveform</div><canvas id="cw" height="60"></canvas>
<div class="mut">Log-mel (Whisper feature extractor, 80 bins, 10 ms hop, cropped) with attribution overlay</div><canvas id="cm" height="160"></canvas>
<div class="mut">AADD v3 timeline (12 classes, 20 ms; estimates on real audio) and flagged frames (any p ≥ 0.5)</div><canvas id="ca" height="110"></canvas>
<div class="mut">Sentiment importance over time (|IG| of Cardiff's predicted class, projected via word timestamps)</div><canvas id="cs" height="30"></canvas>
<div style="margin-top:8px" class="lg mut"><span style="outline:2px solid var(--sub)">substitution</span><span style="outline:2px dashed var(--ins)">insertion</span><span>shade = sentiment importance (blue supports, red opposes the predicted class)</span></div>
<div id="trans" style="margin-top:6px"></div><div class="mut" id="goldt" style="margin-top:6px"></div>
</div>
<h2>All clips</h2><div class="card"><table id="tbl"></table></div>
<p class="mut">Credits: AADD (MIT, github.com/gertelrina/aadd) for degradation timelines; Captum for LayerIntegratedGradients; Whisper (OpenAI) for ASR and alignment heads; the SPES idea (arXiv:2411.01710) for spectrogram perturbation. No audio is embedded.</p>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const f=(x,d=3)=>x==null||Number.isNaN(x)?'NaN':(+x).toFixed(d);
const ci=o=>o&&o.ci?`${f(o.mean)} [${f(o.ci[0])}, ${f(o.ci[1])}] (n=${o.n})`:'n/a';
const S=D.summary;const ASR=(S.asr||'').split('/').pop();$('ttl').textContent=`AudioXAI: does ${ASR} attend to degraded audio?`;
$('sub').textContent=`${S.slice} slice, n=${S.n_clips} clips, ASR ${S.asr}, sentiment ${S.sentiment}. ${S.aadd}. ${S.aadd_caveat}`;
function card(t,v,body){return `<div class="card"><div class="v">${t}: ${v}</div><div class="mut">${body}</div></div>`}
let h=card('H1 attribution concentrates in flagged audio',S.H1.verdict,`mean focus ratio (M1) ${ci(S.H1)}; median ${f(S.H1.median)}. M3 ${ci(S.focus_ratio_m3)}, M4 ${ci(S.focus_ratio_m4)}. Time-profile agreement Spearman M1-M3 ${ci(S.rho_m1_m3_time)}, M1-M4 ${ci(S.rho_m1_m4_time)}. Clips with zero flagged frames: ${S.n_zero_flagged} (NaN, kept in table); all frames flagged: ${S.n_all_flagged} (ratio 1 by construction). Informative clips (0&lt;flagged&lt;1): M1 ${ci(S.focus_ratio_m1_informative)}, M4 ${ci(S.focus_ratio_m4_informative)}. Graded Spearman(M1 time profile, max AADD prob) ${ci(S.rho_m1_vs_aadd_maxprob)}.`);
if(S.H2){const H=S.H2;h+=card('H2 errors in flagged audio, on low-importance words',H.verdict,`error rate flagged ${f(H.pooled_err_rate_flagged)} (${H.n_ref_words_flagged} ref words) vs clean ${f(H.pooled_err_rate_clean)} (${H.n_ref_words_clean}); paired diff ${ci(H.paired_wer_diff_flagged_minus_clean)}. Importance share of erroneous words minus their word share ${ci(H.err_imp_share_minus_word_share)}. Sentiment importance share in flagged ${ci(H.sent_imp_share_flagged)}.`)}
const h3v=(S.H3.verdict||'').startsWith('pending')&&ASR!='whisper-tiny'?`H3 is decided on whisper-tiny only (no Phase 1 ablation for ${ASR})`:(S.H3.verdict||'');h+=card('H3 text classifier is the bottleneck',h3v,(S.H3.detail||'')+` Sample: Cardiff accuracy on ASR text ${f(S.cardiff_acc_asr_text_sample)}, on gold text ${f(S.cardiff_acc_gold_text_sample)}.`);
$('verd').innerHTML=h;
$('answer').innerHTML=`<b>Reading.</b> ${ASR}'s attribution does not concentrate on AADD-flagged audio (mean focus ratio ${f(S.H1.mean)}, 95% CI [${f(S.H1.ci[0])}, ${f(S.H1.ci[1])}], n=${S.H1.n}; graded rank correlation with AADD probability ${f(S.rho_m1_vs_aadd_maxprob.mean)}). `+
(S.H2?`ASR errors are not more frequent in flagged audio (${f(S.H2.pooled_err_rate_flagged)} vs ${f(S.H2.pooled_err_rate_clean)}) and erroneous words carry about their share of sentiment importance (${f(S.H2.mean_err_word_imp_share)} vs ${f(S.H2.mean_err_word_share)} word share). `:'')+
(S.H3.macro_f1_gold!=null?`Human gold text raises Cardiff macro-F1 only from ${f(S.H3.macro_f1_full)} to ${f(S.H3.macro_f1_gold)}. `:`On this sample Cardiff accuracy is ${f(S.cardiff_acc_asr_text_sample)} on ASR text and ${f(S.cardiff_acc_gold_text_sample)} on gold text. `)+`The small ASR effect is better explained by a classifier ceiling than by where the ASR looks. Caveats: AADD flags are estimates on real audio, and M1 ${S.randomization&&S.randomization.verdict=='FAIL'?'fails':'passes'} the model-randomization check (Spearman ${S.randomization?f(S.randomization.rho_tf_abs.mean):'n/a'}); a failure means its maps are partly driven by the input itself, although deletion shows they still rank patches better than random.`;
let fh='';if(S.deletion){const d=S.deletion;fh+=`<div>Deletion (top-k% M1 patches vs random, k=5..50%, n=${d.n}): AUC top ${ci(d.auc_top)}, random ${ci(d.auc_random)}, diff ${ci(d.auc_diff)}; clips with top &gt; random ${f(d.frac_clips_top_gt_random,2)}. ${d.unit}.</div>`}
if(S.randomization){const r=S.randomization;fh+=`<div>Model randomization (top 2 decoder layers re-initialized, n=${r.n}): Spearman |M1| TF ${ci(r.rho_tf_abs)}, time profile ${ci(r.rho_time_abs)} → <b>${r.verdict}</b> (${r.rule}). Correlation of |M1| with |input|: ${ci(r.rho_map_vs_absinput)}.</div>`}
if(S.m2){fh+=`<div>M1 vs M2 (SPES-style patch occlusion, n=${S.m2.n} clips): median per-token Spearman of time profiles ${ci(S.m2.m1_m2_rho_time_median)}; M2 focus ratio ${ci(S.m2.focus_ratio_m2)}.</div>`}
if(S.e2e_segments){const e=S.e2e_segments;fh+=`<div>End-to-end 500 ms window replacement (n=${e.n} clips): mean |ΔP(gold)| ${ci(e.mean_abs_change)}; in flagged windows ${ci(e.dp_flagged_windows)}, clean windows ${ci(e.dp_clean_windows)}.</div>`}
$('faith').innerHTML=fh||'not run';
(function(){const F=D.faith.filter(r=>r.auc_top!=null);if(!F.length)return;const ks=[5,10,15,20,25,30,35,40,45,50];
const mt=ks.map(k=>F.reduce((a,r)=>a+r['drop_top_'+k],0)/F.length),mr=ks.map(k=>F.reduce((a,r)=>a+r['drop_rand_'+k],0)/F.length);
const svg=$('delc');const W=svg.clientWidth||800,H=160,mx=Math.max(...mt,...mr)*1.1;const X=k=>40+(k-5)/45*(W-60),Y=v=>H-v/mx*(H-15);
const pl=(a,d)=>`<polyline fill="none" stroke="var(--acc)" stroke-width="2" ${d?'stroke-dasharray="5"':''} points="${a.map((v,i)=>X(ks[i])+','+Y(v)).join(' ')}"/>`;
svg.innerHTML=pl(mt,0)+pl(mr,1)+`<line x1="40" x2="${W-20}" y1="${H}" y2="${H}" stroke="var(--line)"/><text x="40" y="${H+16}" font-size="11" fill="var(--mut)">k=5%</text><text x="${W-60}" y="${H+16}" font-size="11" fill="var(--mut)">50%</text><text x="2" y="14" font-size="11" fill="var(--mut)">${f(mx,2)}</text><text x="${X(50)-60}" y="${Y(mt[9])-6}" font-size="11" fill="var(--acc)">top-k</text><text x="${X(50)-60}" y="${Y(mr[9])+14}" font-size="11" fill="var(--mut)">random</text>`})();
// histogram
(function(){const v=D.table.map(r=>r.focus_ratio_m1).filter(x=>x!=null&&isFinite(x));const svg=$('hist');const W=svg.clientWidth||800,H=150;if(!v.length)return;
const mx=Math.max(2,...v);const nb=20;const b=new Array(nb).fill(0);v.forEach(x=>b[Math.min(nb-1,Math.floor(x/mx*nb))]++);const bm=Math.max(...b);
let s='';b.forEach((c,i)=>{const x=30+i*(W-40)/nb,hh=c/bm*(H-25);s+=`<rect x="${x}" y="${H-hh}" width="${(W-40)/nb-2}" height="${hh}" fill="var(--acc)"><title>${c}</title></rect>`});
const x1=30+1/mx*(W-40);s+=`<line x1="${x1}" x2="${x1}" y1="5" y2="${H}" stroke="var(--bad)" stroke-dasharray="4"/><text x="${x1+4}" y="14" fill="var(--bad)" font-size="12">1</text><text x="${W-60}" y="${H+15}" fill="var(--mut)" font-size="12">${f(mx,1)}</text><text x="30" y="${H+15}" fill="var(--mut)" font-size="12">0</text>`;svg.innerHTML=s})();
function dec(q){const s=atob(q.b);const a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a}
function cmap(t){t=Math.max(0,Math.min(1,t));return [Math.round(255*Math.min(1,1.5*t)),Math.round(255*Math.max(0,t*1.6-.6)),Math.round(255*(.25+.5*(1-t)*t))]}
function fit(c){const w=c.clientWidth||900;c.width=w;return w}
let C=null;
const cs=$('clipSel');D.clips.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${c.file} (${c.label}→${c.pred})`;cs.appendChild(o)});
cs.onchange=()=>{C=D.clips[+cs.value];fillTok();draw()};$('meth').onchange=draw;$('tokSel').onchange=draw;
function fillTok(){const t=$('tokSel');t.innerHTML='<option value="0">mean over tokens</option>'+C.toks.map((s,i)=>`<option value="${i+1}">${i}: ${s.replace(/</g,'&lt;')}</option>`).join('')}
function draw(){if(!C)return;const dur=C.dur;
$('clipHead').innerHTML=`<b>${C.file}</b> gold label <b>${C.label}</b>, Cardiff on ASR text <b>${C.pred}</b> (p=${C.probs.join('/')}), on gold text <b>${C.pred_gold}</b>, WER ${f((C.cnt.S+C.cnt.D+C.cnt.I)/Math.max(1,C.cnt.N))} (S${C.cnt.S} D${C.cnt.D} I${C.cnt.I} N${C.cnt.N}), ${f(dur,2)} s`;
let c=$('cw'),w=fit(c),g=c.getContext('2d');g.clearRect(0,0,w,60);g.fillStyle=getComputedStyle(document.body).color;C.env.forEach((e,i)=>{const x=i/C.env.length*w;g.fillRect(x,30-e[1]*28,Math.max(1,w/C.env.length),Math.max(1,(e[1]-e[0])*28))});
c=$('cm');w=fit(c);g=c.getContext('2d');const mel=dec(C.mel),[nb,nt]=C.mel.shape;const img=g.createImageData(nt,nb);
const meth=$('meth').value,tk=+$('tokSel').value;let ov=null,ovs=null;
if(meth=='m1'&&C.m1.shape[0]){ov=dec(C.m1);ovs=C.m1.shape}if(meth=='m4'&&C.m4.shape[0]){ov=dec(C.m4);ovs=C.m4.shape}if(meth=='m3'&&C.m3&&C.m3.shape[0]){ov=dec(C.m3);ovs=C.m3.shape}
for(let y=0;y<nb;y++)for(let x=0;x<nt;x++){const v=mel[y*nt+x]/255;let r=v*200,gg=v*200,b=v*200;
 if(ov){let a;if(meth=='m1'){const band=Math.floor(y/8);a=ov[(tk*ovs[1]+band)*ovs[2]+x]/255}else{a=ov[tk*ovs[1]+x]/255}const cc=cmap(a);const al=Math.min(1,a*1.2);r=r*(1-al)+cc[0]*al;gg=gg*(1-al)+cc[1]*al;b=b*(1-al)+cc[2]*al}
 const i=((nb-1-y)*nt+x)*4;img.data[i]=r;img.data[i+1]=gg;img.data[i+2]=b;img.data[i+3]=255}
const off=document.createElement('canvas');off.width=nt;off.height=nb;off.getContext('2d').putImageData(img,0,0);g.imageSmoothingEnabled=false;g.drawImage(off,0,0,w,160);
if(tk>0){const wd=C.words;}
c=$('ca');w=fit(c);g=c.getContext('2d');g.clearRect(0,0,w,110);const A=dec(C.aadd),[na,ta]=C.aadd.shape;const rh=7;
for(let k=0;k<na;k++)for(let x=0;x<ta;x++){const v=A[k*ta+x]/255*(C.aadd.hi-C.aadd.lo)+C.aadd.lo;if(v<0.02)continue;g.fillStyle=`rgba(200,60,40,${Math.min(1,v)})`;g.fillRect(x/ta*w,k*rh,w/ta+1,rh)}
g.fillStyle=getComputedStyle(document.body).color;g.font='9px sans-serif';D.classes.forEach((n,k)=>g.fillText(n,2,k*rh+6));
C.flag.forEach((v,x)=>{if(v>0){g.fillStyle=`rgba(230,120,0,${v})`;g.fillRect(x/C.flag.length*w,92,w/C.flag.length+1,14)}});g.fillStyle=getComputedStyle(document.body).color;g.fillText('flagged',2,103);
c=$('cs');w=fit(c);g=c.getContext('2d');g.clearRect(0,0,w,30);C.words.forEach(o=>{g.fillStyle=o.sgn>0?`rgba(47,111,176,${o.imp})`:`rgba(192,57,43,${o.imp})`;g.fillRect(o.t0/dur*w,0,Math.max(2,(o.t1-o.t0)/dur*w),30)});
$('trans').innerHTML='ASR: '+C.words.map(o=>`<span class="w tag-${o.tag}" title="${o.t0}-${o.t1}s ${o.tag} imp ${o.imp}" style="background:${o.sgn>0?'rgba(47,111,176,':'rgba(192,57,43,'}${o.imp*0.8})">${o.w.replace(/</g,'&lt;')}</span>`).join(' ');
$('goldt').textContent='Gold (SLUE normalized_text): '+C.gold}
// table
const cols=D.table.length?Object.keys(D.table[0]):[];let sortK=null,asc=true;
function tbl(){let r=[...D.table];if(sortK)r.sort((a,b)=>((a[sortK]??-1e9)>(b[sortK]??-1e9)?1:-1)*(asc?1:-1));
$('tbl').innerHTML='<tr>'+cols.map(c=>`<th data-k="${c}">${c}</th>`).join('')+'</tr>'+r.map(o=>'<tr>'+cols.map(c=>`<td>${o[c]==null?'NaN':o[c]}</td>`).join('')+'</tr>').join('');
document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;asc=sortK==k?!asc:true;sortK=k;tbl()})}
tbl();if(D.clips.length){C=D.clips[0];fillTok();draw()}window.onresize=draw;
</script></body></html>"""
