"""
qid_multi.py - perluasan multibahasa.
  (1) NusaX sentimen paralel dalam 12 bahasa: Indonesia, Inggris, dan 10 bahasa daerah
      (Jawa, Sunda, Bali, Minangkabau, Banjar, Madura, Bugis, Aceh, Ngaju, Batak Toba).
      Kalimatnya sama (terjemahan), sehingga perbedaan skor hanya berasal dari bahasa.
  (2) TyDiQA-GoldP bahasa Inggris sebagai pembanding bahasa bersumber daya besar.

    python qid_multi.py --model qwen1.5b --prec bf16
Hasil: results/multi_{model}_{prec}.json  (bisa dihentikan lalu dilanjutkan; tugas selesai dilewati)
"""
import argparse, json, urllib.request
from pathlib import Path

import torch
import qid_eval as E

LANGS = ["indonesian", "english", "javanese", "sundanese", "balinese", "minangkabau", "banjarese", "madurese",
         "buginese", "acehnese", "ngaju", "toba_batak"]
TY_EN = E.DATA / "tydiqa_en_dev.json"


def nusax_lang(model, tok, lang):
    items = json.loads((E.DATA / "nusax_all" / f"{lang}.json").read_text(encoding="utf8"))
    labs = list(E.SENT.values()); per = []
    for it in items:
        pre = E.chat(tok, "Tentukan sentimen kalimat berikut. Jawab dengan satu kata: positif, negatif, atau netral.\n\n"
                          f"Kalimat: {it['text']}") + "Sentimen:"
        sc = E.seq_logprob(model, tok, pre, [" " + l for l in labs])
        pred = labs[max(range(3), key=lambda i: sc[i])]
        per.append(dict(id=it["id"], pred=pred, gold=E.SENT[it["label"]], ok=int(pred == E.SENT[it["label"]])))
    acc = sum(p["ok"] for p in per) / len(per); f1s = []
    for c in labs:
        tp = sum(p["pred"] == c and p["gold"] == c for p in per); fp = sum(p["pred"] == c and p["gold"] != c for p in per)
        fn = sum(p["pred"] != c and p["gold"] == c for p in per)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return dict(acc=acc, macro_f1=sum(f1s) / 3, n=len(per)), per


def get_tydiqa_en():
    if TY_EN.exists(): return
    try:
        with urllib.request.urlopen("https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json", timeout=180) as r:
            d = json.loads(r.read().decode("utf8"))
        items = [dict(id=q["id"], context=p["context"], question=q["question"], answers=sorted({a["text"] for a in q["answers"]}))
                 for art in d["data"] for p in art["paragraphs"] for q in p["qas"] if q["id"].lower().startswith("english")]
    except Exception as e:
        print("[info] unduhan langsung gagal (", e, "), mencoba Hugging Face ...")
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
        items = [dict(id=r["id"], context=r["context"], question=r["question"], answers=sorted(set(r["answers"]["text"])))
                 for r in ds if r["id"].lower().startswith("english")]
    TY_EN.write_text(json.dumps(items, ensure_ascii=False), encoding="utf8")
    print(f"[OK] {TY_EN}: {len(items)} pertanyaan bahasa Inggris")


def tydiqa_en(model, tok, bs):
    items = json.loads(TY_EN.read_text(encoding="utf8")); texts = []
    for it in items:
        ctx = tok.convert_tokens_to_string(tok.tokenize(it["context"])[:1200])
        texts.append(E.chat(tok, "Answer the question using the passage below. The answer must be a short quote from the passage, "
                                 f"without explanation.\n\nPassage: {ctx}\n\nQuestion: {it['question']}") + "Answer:")
    outs = E.generate(model, tok, texts, bs, 48); per = []
    for it, o in zip(items, outs):
        o = o.strip().split("\n")[0]
        per.append(dict(id=it["id"], pred=o, em=max(float(E.norm(o) == E.norm(a)) for a in it["answers"]),
                        f1=max(E.f1_tok(o, a) for a in it["answers"])))
    return dict(em=sum(p["em"] for p in per) / len(per), f1=sum(p["f1"] for p in per) / len(per), n=len(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(E.MODELS), required=True)
    ap.add_argument("--prec", choices=["bf16", "int8", "nf4"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    E.OUT.mkdir(exist_ok=True)
    try: get_tydiqa_en()
    except Exception as e: print("[info] TyDiQA Inggris tidak bisa diunduh, dilewati:", e)
    path = E.OUT / f"multi_{a.model}_{a.prec}.json"
    res = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
    tasks = [f"nusax_{l}" for l in LANGS] + (["tydiqa_en"] if TY_EN.exists() else [])
    todo = [t for t in tasks if not (t in res and not res[t].get("interrupted", False))]
    if not todo: print(f"[lewati] {a.model} {a.prec}: semua tugas multibahasa selesai"); return
    tok, model, info = E.load(a.model, a.prec)
    res.update(model=a.model, prec=a.prec, hf_id=info["hf_id"])
    for t in todo:
        E.reset()
        with E.PowerMeter() as pm:
            m, per = tydiqa_en(model, tok, a.batch) if t == "tydiqa_en" else nusax_lang(model, tok, t[6:])
        m.update(sec=round(pm.sec, 1), energy_wh=round(pm.wh, 3), interrupted=pm.interrupted)
        res[t] = m; res[t + "_items"] = per
        path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf8")
        print(f"[OK] {a.model} {a.prec} {t}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float) and k in ("acc", "macro_f1", "em", "f1")))
    print("hasil ->", path)


if __name__ == "__main__":
    main()
