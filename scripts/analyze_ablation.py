"""Ablation analysis under the final protocol. 'full' = LiteMedSeg rows of results/final_protocol.
Tier 1: removal / replacement / HGA decomposition.  Tier 2: single-module additions to the bare network + all-standard.
Matched: width-matched controls (Abl-bare_b20, Abl-wo_hga_none_b16). Output: results/final_protocol_ablation/ablation_table.csv"""
import csv, collections, statistics as st
from scipy import stats
OUT = "results/final_protocol_ablation"
rows = list(csv.DictReader(open(f"{OUT}/campaign_summary.csv")))
rows += [x for x in csv.DictReader(open("results/final_protocol/campaign_summary.csv")) if x["model"] == "LiteMedSeg"]
D = collections.defaultdict(dict); P = {}
for x in rows:
    if x["status"] != "ok": continue
    if x["model"] == "LiteMedSeg": v = "full"
    else:
        v = x["model"][4:]
        v = v[:-4] if v.endswith("_b12") else v           # keep _b16/_b20 suffix for matched controls
    D[(x["dataset"], v)][int(x["seed"])] = {k: float(x[k]) for k in ("dice", "iou", "precision", "recall")}
    P[v] = int(x["params"])
dsets = ["CVC-ClinicDB", "Kvasir-SEG", "PolypGen"]
groups = [
 ("Reference", ["full", "bare"]),
 ("Module removed (vs full)", ["wo_lsk_none", "wo_eca_none", "wo_mpp_none", "wo_hga_none"]),
 ("Module replaced by standard (vs full)", ["wo_lsk_std", "wo_eca_std", "wo_mpp_std"]),
 ("HGA decomposition (vs full)", ["hga_gate_only", "hga_attn_only", "hga_std_gate"]),
 ("Single module added to bare (vs bare)", ["bare_lsk", "bare_eca", "bare_mpp", "bare_hga"]),
 ("All-standard network", ["baseline_std"]),
 ("Width-matched controls (vs full)", ["bare_b20", "wo_hga_none_b16"]),
]
ms = lambda a: (st.mean(a), st.stdev(a) if len(a) > 1 else 0.0)
ref_for = lambda g: "bare" if g.startswith("Single module") else "full"
lines = ["group,variant,params,n_seeds_min," + ",".join(f"{d}_dice_mean,{d}_dice_std,{d}_dDice,{d}_p" for d in dsets) + ",avg_dice,avg_dDice,reference"]
text = []
for g, vs in groups:
    ref = ref_for(g)
    for v in vs:
        if not all(len(D.get((d, v), {})) for d in dsets):
            continue
        r = [g, v, P[v]]; avgs, davgs, nmin = [], [], 9
        cells = []
        for d in dsets:
            a, f = D[(d, v)], D[(d, ref)]
            seeds = sorted(set(a) & set(f)) if v != ref else sorted(a)
            nmin = min(nmin, len(seeds))
            m, s = ms([a[k]["dice"] for k in seeds])
            diffs = [a[k]["dice"] - f[k]["dice"] for k in seeds]
            p = stats.ttest_rel([a[k]["dice"] for k in seeds], [f[k]["dice"] for k in seeds]).pvalue if v != ref and len(seeds) > 2 else float("nan")
            cells += [f"{m:.4f}", f"{s:.4f}", f"{st.mean(diffs):+.4f}", f"{p:.3f}"]; avgs.append(m); davgs.append(st.mean(diffs))
        r = r[:3] + [nmin] + cells + [f"{st.mean(avgs):.4f}", f"{st.mean(davgs):+.4f}", ref]
        lines.append(",".join(map(str, r)))
        text.append((g, v, P[v], nmin, cells, st.mean(avgs), st.mean(davgs), ref))
open(f"{OUT}/ablation_table.csv", "w").write("\n".join(lines) + "\n")
cur = None
for g, v, p, n, c, a, da, ref in text:
    if g != cur: print("\n##", g); cur = g
    print(v.ljust(16), f"{p:>10,}", f"n={n}", *[f"{c[4*i]}±{c[4*i+1]} ({c[4*i+2]},p={c[4*i+3]})" for i in range(3)], f"| avg {a:.3f} d={da:+.3f} vs {ref}")
