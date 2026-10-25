"""
qid_report.py - rangkum semua results/*.json menjadi tabel (CSV) dan grafik.

    python qid_report.py
Keluaran: report/ringkasan.csv, report/per_kelompok.csv, report/per_jenjang.csv, report/fig_*.png
"""
import json
from pathlib import Path

import pandas as pd

R, OUT = Path("results"), Path("report")
ORDER = {"bf16": 0, "int8": 1, "nf4": 2}


def main():
    OUT.mkdir(exist_ok=True)
    rows, grp, lvl = [], [], []
    apps = []
    for p in sorted(R.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf8"))
        if p.name.startswith("app_") or "model" not in r:
            if "tag" in r: apps.append({k: v for k, v in r.items() if k != "items"})
            continue
        row = dict(model=r["model"], prec=r["prec"], weights_gb=r.get("weights_gb"), load_sec=r.get("load_sec"))
        for t, keys in [("indommlu", ["acc"]), ("nusax", ["acc", "macro_f1"]), ("tydiqa", ["em", "f1"]),
                        ("speed", ["ttft_s", "tokens_per_s"])]:
            if t not in r: continue
            for k in keys: row[f"{t}_{k}"] = r[t][k]
            row[f"{t}_peak_vram_gb"] = r[t]["peak_vram_gb"]; row[f"{t}_energy_wh"] = r[t]["energy_wh"]; row[f"{t}_sec"] = r[t]["sec"]
            if t != "speed": row[f"{t}_wh_per_1000"] = r[t]["energy_wh"] / r[t]["n"] * 1000
        if "speed" in r and r["speed"].get("energy_wh") == r["speed"].get("energy_wh"):
            row["speed_wh_per_1000_tokens"] = r["speed"]["energy_wh"] / (r["speed"]["n"] * r["speed"]["new_tokens"]) * 1000
        rows.append(row)
        if "indommlu" in r:
            for g, v in r["indommlu"]["by_group"].items(): grp.append(dict(model=r["model"], prec=r["prec"], group=g, acc=v))
            for g, v in r["indommlu"]["by_level"].items(): lvl.append(dict(model=r["model"], prec=r["prec"], level=g, acc=v))
    df = pd.DataFrame(rows)
    df["o"] = df.prec.map(ORDER); df = df.sort_values(["model", "o"]).drop(columns="o")
    # retensi kualitas relatif terhadap bf16 model yang sama
    for c in ["indommlu_acc", "nusax_macro_f1", "tydiqa_f1"]:
        if c in df:
            base = df[df.prec == "bf16"].set_index("model")[c]
            df[c + "_vs_bf16"] = df.apply(lambda r: r[c] / base.get(r["model"], float("nan")), axis=1)
    df.to_csv(OUT / "ringkasan.csv", index=False)
    if apps:
        pd.DataFrame(apps).to_csv(OUT / "purwarupa.csv", index=False)
        print(pd.DataFrame(apps).round(3).to_string(index=False))
    pd.DataFrame(grp).pivot_table(index=["model", "prec"], columns="group", values="acc").to_csv(OUT / "per_kelompok.csv")
    pd.DataFrame(lvl).pivot_table(index=["model", "prec"], columns="level", values="acc").to_csv(OUT / "per_jenjang.csv")
    print(df.round(3).to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        mk = {"bf16": "o", "int8": "s", "nf4": "^"}
        for y, lab in [("indommlu_acc", "IndoMMLU accuracy"), ("tydiqa_f1", "TyDiQA-ID F1"), ("nusax_macro_f1", "NusaX macro-F1")]:
            if y not in df: continue
            fig, ax = plt.subplots(figsize=(5, 3.6))
            for m, sub in df.groupby("model"):
                ax.plot(sub["indommlu_peak_vram_gb"], sub[y], "-", lw=1, alpha=.5)
                for _, r in sub.iterrows():
                    ax.scatter(r["indommlu_peak_vram_gb"], r[y], marker=mk[r["prec"]], s=40)
                    ax.annotate(f"{m} {r['prec']}", (r["indommlu_peak_vram_gb"], r[y]), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Peak GPU memory (GB)"); ax.set_ylabel(lab); fig.tight_layout()
            fig.savefig(OUT / f"fig_{y}.png", dpi=200); plt.close(fig)
        print("grafik ->", OUT)
    except Exception as e:
        print("[info] grafik dilewati:", e)


if __name__ == "__main__":
    main()
