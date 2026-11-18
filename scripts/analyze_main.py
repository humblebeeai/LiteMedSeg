"""Main comparison under the final protocol: Table 3 (macro metrics, mean±std over 3 training seeds,
one fixed split) + paired tests of LiteMedSeg vs every baseline (per dataset, Holm-corrected)."""
import csv, collections, statistics as st
from scipy import stats
OUT = "results/final_protocol"
NAME = {"UNet": "U-Net", "UNet++": "UNet++", "ResUNet": "ResUNet", "ResUNet++": "ResUNet++", "AttentionUNet": "Attention U-Net",
        "DeepLabV3+": "DeepLabV3+", "SCA-InceptionUNet": "SCA-InceptionUNeXt", "UNeXt": "UNeXt", "Fast-SCNN": "Fast-SCNN",
        "MobileViT-Seg": "MobileViT-Seg", "LiteMedSeg": "LiteMedSeg (ours)"}
ORDER = list(NAME)
DS = ["CVC-ClinicDB", "Kvasir-SEG", "PolypGen"]
M = ["precision", "recall", "f1", "iou", "dice"]
rows = [x for x in csv.DictReader(open(f"{OUT}/campaign_summary.csv")) if x["status"] == "ok"]
D = collections.defaultdict(dict); INFO = {}
for x in rows:
    D[(x["dataset"], x["model"])][int(x["seed"])] = {k: float(x[k]) for k in M + ["fps"]}
    INFO[x["model"]] = (int(x["params"]), float(x["gflops"]))
for d in DS:
    for m in ORDER: assert len(D[(d, m)]) == 3, (d, m)
f = lambda a: f"{st.mean(a):.3f} ± {st.stdev(a):.3f}"
t3 = [["Dataset", "Model"] + ["Precision", "Recall", "F1", "IoU", "Dice"] + ["Params(M)", "GFLOPs"]]
for d in DS:
    for m in ORDER:
        t3.append([d, NAME[m]] + [f([D[(d, m)][s][k] for s in range(3)]) for k in M] + [f"{INFO[m][0]/1e6:.3f}", f"{INFO[m][1]:.2f}"])
csv.writer(open(f"{OUT}/table3_main.csv", "w", newline="", encoding="utf-8")).writerows(t3)
# paired tests
tq = stats.t.ppf(0.975, 2); sig = [["Dataset", "Baseline", "mean_dDice", "CI95_lo", "CI95_hi", "p_paired_t", "p_Holm", "dIoU", "wins_of_3"]]
for d in DS:
    res = []
    for m in ORDER[:-1]:
        a = [D[(d, "LiteMedSeg")][s]["dice"] for s in range(3)]; b = [D[(d, m)][s]["dice"] for s in range(3)]
        df = [x - y for x, y in zip(a, b)]; mu = st.mean(df); se = st.stdev(df) / 3 ** 0.5
        p = stats.ttest_rel(a, b).pvalue
        di = st.mean(D[(d, "LiteMedSeg")][s]["iou"] - D[(d, m)][s]["iou"] for s in range(3))
        res.append([m, mu, mu - tq * se, mu + tq * se, p, di, sum(x > 0 for x in df)])
    order = sorted(range(len(res)), key=lambda i: res[i][4]); holm = [0] * len(res); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, res[i][4] * (len(res) - r))); holm[i] = run
    for i, r in enumerate(res):
        sig.append([d, NAME[r[0]], f"{r[1]:+.4f}", f"{r[2]:+.4f}", f"{r[3]:+.4f}", f"{r[4]:.3f}", f"{holm[i]:.3f}", f"{r[5]:+.4f}", f"{r[6]}/3"])
csv.writer(open(f"{OUT}/significance_main.csv", "w", newline="", encoding="utf-8")).writerows(sig)
# console summary (ascii)
for d in DS:
    print("==", d)
    for m in ORDER:
        print(NAME[m].ljust(20), " ".join(f([D[(d, m)][s][k] for s in range(3)]) for k in ("dice", "iou")), f"{INFO[m][0]/1e6:.2f}M")
print()
for r in sig[1:]: print(*r, sep=" | ")
