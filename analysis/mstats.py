import json, numpy as np
from scipy.stats import binomtest, spearmanr
R = "/home/claude/quantid_res2/r/"
rng = np.random.default_rng(7)
L = ["indonesian", "english", "javanese", "sundanese", "balinese", "minangkabau", "banjarese", "madurese", "buginese", "acehnese", "ngaju", "toba_batak"]
LOCAL = L[2:]
CFG = [("gemma1b", p) for p in ("bf16", "int8", "nf4")] + [("qwen1.5b", p) for p in ("bf16", "int8", "nf4")] + [("qwen3b", p) for p in ("bf16", "int8", "nf4")] + [("qwen7b", "nf4")]
M = {f"{m}_{p}": json.load(open(f"{R}multi_{m}_{p}.json")) for m, p in CFG}
out = {"acc": {}, "f1": {}, "tax": {}, "ty_en": {}, "words_en": {}, "agg": {}}
for k, r in M.items():
    out["acc"][k] = {l: r[f"nusax_{l}"]["acc"] for l in L}; out["f1"][k] = {l: r[f"nusax_{l}"]["macro_f1"] for l in L}
    out["ty_en"][k] = dict(f1=r["tydiqa_en"]["f1"], em=r["tydiqa_en"]["em"], n=r["tydiqa_en"]["n"])
    out["words_en"][k] = float(np.mean([len(i["pred"].split()) for i in r["tydiqa_en_items"]]))
def mc(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    return n01, n10, (binomtest(min(n01, n10), n01 + n10, .5).pvalue if n01 + n10 else 1.0)
tests = []
for m in ("gemma1b", "qwen1.5b", "qwen3b"):
    for q in ("int8", "nf4"):
        A, B = M[f"{m}_bf16"], M[f"{m}_{q}"]
        for l in L:
            a = np.array([i["ok"] for i in A[f"nusax_{l}_items"]]); b = np.array([i["ok"] for i in B[f"nusax_{l}_items"]])
            n01, n10, p = mc(a, b); tests.append(dict(model=m, prec=q, lang=l, diff=float((b - a).mean()), n01=n01, n10=n10, p=float(p)))
        # pooled local-language tax with bootstrap over sentences (same ids across languages)
        da = {l: np.array([i["ok"] for i in B[f"nusax_{l}_items"]]) - np.array([i["ok"] for i in A[f"nusax_{l}_items"]]) for l in L}
        loc = np.mean([da[l] for l in LOCAL], axis=0); hi = np.mean([da[l] for l in ("indonesian", "english")], axis=0)
        gap = loc - hi; bs = np.array([gap[rng.integers(0, len(gap), len(gap))].mean() for _ in range(5000)])
        out["agg"][f"{m}_{q}"] = dict(local=float(loc.mean()), high=float(hi.mean()), gap=float(gap.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                     p=float(2 * min((bs >= 0).mean(), (bs <= 0).mean())))
ps = np.array([t["p"] for t in tests]); o = np.argsort(ps); run = 0
for r_, i in enumerate(o): run = max(run, (len(ps) - r_) * ps[i]); tests[i]["p_holm"] = float(min(1, run))
out["tests"] = tests
# TyDiQA EN paired bootstrap bf16 vs nf4
tyt = []
for m in ("gemma1b", "qwen1.5b", "qwen3b"):
    for q in ("int8", "nf4"):
        a = np.array([i["f1"] for i in M[f"{m}_bf16"]["tydiqa_en_items"]]); b = np.array([i["f1"] for i in M[f"{m}_{q}"]["tydiqa_en_items"]]); d = b - a
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        tyt.append(dict(model=m, prec=q, diff=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p=float(max(1 / 5000, 2 * min((bs >= 0).mean(), (bs <= 0).mean())))))
out["ty_tests"] = tyt
# chance & majority
import collections
labs = [i["gold"] for i in M["gemma1b_bf16"]["nusax_indonesian_items"]]; out["majority"] = max(collections.Counter(labs).values()) / len(labs)
json.dump(out, open("mstats.json", "w"), indent=1)
print("majority", out["majority"])
for k, v in out["agg"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 4) for x in b]) for a, b in v.items()})
sig = [t for t in tests if t["p_holm"] < .05]; print("sig", len(sig), [(t["model"], t["prec"], t["lang"], round(t["diff"], 3)) for t in sig])
for t in tyt: print(t)
for k in out["words_en"]: print(k, round(out["words_en"][k], 1), round(out["ty_en"][k]["f1"], 3), round(out["ty_en"][k]["em"], 3))
# gap ind - mean local per config (bf16)
for k in out["acc"]: print(k, "ind-local gap", round(out["acc"][k]["indonesian"] - np.mean([out["acc"][k][l] for l in LOCAL]), 3), "local mean", round(np.mean([out["acc"][k][l] for l in LOCAL]), 3))

# ---- flip rate and prior shift vs. competence
import collections
flip = {}
for m in ("gemma1b", "qwen1.5b", "qwen3b"):
    for q in ("int8", "nf4"):
        rows = []
        for l in L:
            a = [i["pred"] for i in M[f"{m}_bf16"][f"nusax_{l}_items"]]; b = [i["pred"] for i in M[f"{m}_{q}"][f"nusax_{l}_items"]]
            fr = float(np.mean([x != y for x, y in zip(a, b)]))
            ca, cb = collections.Counter(a), collections.Counter(b)
            tvd = 0.5 * sum(abs(ca[c] - cb[c]) for c in ("positif", "negatif", "netral")) / len(a)
            rows.append(dict(lang=l, flip=fr, tvd=float(tvd), acc_bf16=out["acc"][f"{m}_bf16"][l]))
        rho = spearmanr([r["acc_bf16"] for r in rows], [r["flip"] for r in rows])
        flip[f"{m}_{q}"] = dict(rows=rows, rho=float(rho[0]), p=float(rho[1]))
out["flip"] = flip
# pooled correlation across the 3 models (nf4)
xs = [r["acc_bf16"] for k in ("gemma1b_nf4", "qwen1.5b_nf4", "qwen3b_nf4") for r in flip[k]["rows"]]
ys = [r["flip"] for k in ("gemma1b_nf4", "qwen1.5b_nf4", "qwen3b_nf4") for r in flip[k]["rows"]]
rr = spearmanr(xs, ys); out["flip_pooled_nf4"] = dict(rho=float(rr[0]), p=float(rr[1]), n=len(xs))
xs8 = [r["acc_bf16"] for k in ("gemma1b_int8", "qwen1.5b_int8", "qwen3b_int8") for r in flip[k]["rows"]]
ys8 = [r["flip"] for k in ("gemma1b_int8", "qwen1.5b_int8", "qwen3b_int8") for r in flip[k]["rows"]]
rr8 = spearmanr(xs8, ys8); out["flip_pooled_int8"] = dict(rho=float(rr8[0]), p=float(rr8[1]), n=len(xs8))
json.dump(out, open("mstats.json", "w"), indent=1)
for k, v in flip.items(): print(k, round(v["rho"], 2), round(v["p"], 3), [(r["lang"][:3], round(r["flip"], 2)) for r in v["rows"]])
print(out["flip_pooled_nf4"], out["flip_pooled_int8"])
