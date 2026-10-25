"""
qid_eval.py - evaluasi model bahasa kecil pada tugas berbahasa Indonesia dengan presisi bf16 / int8 / nf4,
sekaligus mengukur efisiensi (memori GPU, kecepatan, energi).

Tugas:
  indommlu : 1.200 soal pilihan ganda kurikulum Indonesia (SD-SMP-SMA-Seleksi PTN), skor = akurasi
  nusax    : 400 kalimat analisis sentimen (positif/negatif/netral), skor = akurasi dan macro-F1
  tydiqa   : tanya-jawab bacaan TyDiQA-GoldP bahasa Indonesia, skor = exact match dan F1 token
  speed    : 30 prompt, masing-masing menghasilkan 128 token; mengukur waktu token pertama, token/detik, energi

Contoh:
  python qid_eval.py --model qwen1.5b --prec bf16
  python qid_eval.py --model qwen7b --prec nf4
  python qid_eval.py --model qwen3b --prec int8 --tasks indommlu nusax
Hasil: results/{model}_{prec}.json
"""
import argparse, json, math, os, re, string, threading, time
from collections import Counter
from pathlib import Path

import torch

CUDA = torch.cuda.is_available()
sync = (lambda: torch.cuda.synchronize()) if CUDA else (lambda: None)
peak = (lambda: torch.cuda.max_memory_allocated() / 1e9) if CUDA else (lambda: float("nan"))
reset = (lambda: torch.cuda.reset_peak_memory_stats()) if CUDA else (lambda: None)

MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma1b": "unsloth/gemma-3-1b-it",  # salinan terbuka (tidak gated) dari google/gemma-3-1b-it
}
DATA = Path("data"); OUT = Path("results")
LETTERS = ["A", "B", "C", "D", "E"]
SENT = {"positive": "positif", "negative": "negatif", "neutral": "netral"}


# ------------------------------------------------------------------ energi GPU (NVML)
class PowerMeter:
    """Mencatat daya papan GPU setiap 0,1 detik; energi = integral daya terhadap waktu (Wh)."""
    def __init__(self):
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit(); self.nv = pynvml; self.h = pynvml.nvmlDeviceGetHandleByIndex(0); self.ok = True
        except Exception as e:
            print("[peringatan] NVML tidak tersedia, energi tidak diukur:", e)

    def __enter__(self):
        self.samples, self.ticks, self.stop = [], [], False
        def run():
            while not self.stop:
                now = time.time(); self.ticks.append(now)
                if self.ok:
                    try: self.samples.append((now, self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0))
                    except Exception: pass
                time.sleep(0.1)
        self.t = threading.Thread(target=run, daemon=True); self.t.start()
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        self.stop = True; self.sec = time.time() - self.t0
        self.t.join()
        tk = self.ticks
        self.max_gap = max((tk[i + 1] - tk[i] for i in range(len(tk) - 1)), default=0.0)
        self.interrupted = self.max_gap > 5.0   # jeda >5 detik: laptop tidur/terhenti, waktu & energi tidak sah
        s = self.samples
        self.wh = sum((s[i + 1][0] - s[i][0]) * (s[i][1] + s[i + 1][1]) / 2 for i in range(len(s) - 1)) / 3600 if len(s) > 1 else float("nan")
        self.mean_w = sum(p for _, p in s) / len(s) if s else float("nan")


# ------------------------------------------------------------------ model
def load(name, prec):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    hf = os.environ.get("QID_LOCAL") or MODELS[name]
    tok = AutoTokenizer.from_pretrained(hf)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = dict(device_map="cuda:0" if CUDA else "cpu")
    if "gemma" in name: kw["attn_implementation"] = "eager"
    if prec == "bf16":
        kw["torch_dtype"] = torch.bfloat16
    elif prec == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif prec == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    reset()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(hf, **kw).eval()
    return tok, model, dict(load_sec=round(time.time() - t0, 1), weights_gb=round(torch.cuda.memory_allocated() / 1e9, 3) if CUDA else float("nan"), hf_id=hf)


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def next_logprobs(model, tok, texts, bs):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        lg = model(**enc).logits[:, -1, :].float().log_softmax(-1)
        out.extend(lg.cpu())
    return out


@torch.no_grad()
def seq_logprob(model, tok, prefix, conts):
    """log p(lanjutan | prefix) untuk beberapa lanjutan sekaligus."""
    p = tok(prefix, add_special_tokens=False).input_ids
    ids = [p + tok(c, add_special_tokens=False).input_ids for c in conts]
    L = max(map(len, ids)); pad = tok.pad_token_id
    x = torch.tensor([[pad] * (L - len(s)) + s for s in ids], device=model.device)
    att = (torch.arange(L, device=model.device)[None, :] >= torch.tensor([L - len(s) for s in ids], device=model.device)[:, None]).long()
    lp = model(input_ids=x, attention_mask=att).logits.float().log_softmax(-1)
    res = []
    for k, s in enumerate(ids):
        off = L - len(s); n = len(s) - len(p)
        res.append(sum(float(lp[k, off + len(p) - 1 + j, s[len(p) + j]]) for j in range(n)))
    return res


# ------------------------------------------------------------------ tugas
def run_indommlu(model, tok, bs):
    items = json.loads((DATA / "indommlu_eval.json").read_text(encoding="utf8"))
    texts = [chat(tok, f"Ini adalah soal {it['subject']} untuk {it['level']}. Pilihlah satu jawaban yang benar. "
                       f"Jawab hanya dengan satu huruf.\n\n{it['question']}\n" + "\n".join(it["options"])) + "Jawaban:"
             for it in items]
    cand = {L: sorted({tok(L, add_special_tokens=False).input_ids[0], tok(" " + L, add_special_tokens=False).input_ids[0]}) for L in LETTERS}
    lps = next_logprobs(model, tok, texts, bs)
    per = []
    for it, lp in zip(items, lps):
        avail = [o[0] for o in it["options"]]
        sc = {L: max(float(lp[t]) for t in cand[L]) for L in avail}
        pred = max(sc, key=sc.get)
        per.append(dict(id=it["id"], group=it["group"], level=it["level"], pred=pred, gold=it["answer"], ok=int(pred == it["answer"])))
    acc = sum(p["ok"] for p in per) / len(per)
    by = lambda key: {g: round(sum(p["ok"] for p in per if p[key] == g) / max(1, sum(1 for p in per if p[key] == g)), 4)
                      for g in sorted({p[key] for p in per})}
    return dict(acc=acc, by_group=by("group"), by_level=by("level"), n=len(per)), per


def run_nusax(model, tok, bs):
    items = json.loads((DATA / "nusax_test.json").read_text(encoding="utf8"))
    labs = list(SENT.values())
    per = []
    for it in items:
        pre = chat(tok, "Tentukan sentimen kalimat berikut. Jawab dengan satu kata: positif, negatif, atau netral.\n\n"
                        f"Kalimat: {it['text']}") + "Sentimen:"
        sc = seq_logprob(model, tok, pre, [" " + l for l in labs])
        pred = labs[max(range(3), key=lambda i: sc[i])]
        per.append(dict(id=it["id"], pred=pred, gold=SENT[it["label"]], ok=int(pred == SENT[it["label"]])))
    acc = sum(p["ok"] for p in per) / len(per)
    f1s = []
    for c in labs:
        tp = sum(1 for p in per if p["pred"] == c and p["gold"] == c); fp = sum(1 for p in per if p["pred"] == c and p["gold"] != c)
        fn = sum(1 for p in per if p["pred"] != c and p["gold"] == c)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return dict(acc=acc, macro_f1=sum(f1s) / 3, n=len(per)), per


def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def f1_tok(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g); ns = sum(common.values())
    if not p or not g: return float(p == g)
    if ns == 0: return 0.0
    pr, rc = ns / len(p), ns / len(g); return 2 * pr * rc / (pr + rc)


@torch.no_grad()
def generate(model, tok, texts, bs, max_new):
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def run_tydiqa(model, tok, bs):
    items = json.loads((DATA / "tydiqa_id_dev.json").read_text(encoding="utf8"))
    texts = []
    for it in items:
        ctx = tok.convert_tokens_to_string(tok.tokenize(it["context"])[:1200])
        texts.append(chat(tok, "Jawablah pertanyaan berdasarkan teks berikut. Jawaban harus berupa kutipan singkat dari teks, "
                               f"tanpa penjelasan.\n\nTeks: {ctx}\n\nPertanyaan: {it['question']}") + "Jawaban:")
    outs = generate(model, tok, texts, bs, 48)
    per = []
    for it, o in zip(items, outs):
        o = o.strip().split("\n")[0]
        em = max(float(norm(o) == norm(a)) for a in it["answers"]); f1 = max(f1_tok(o, a) for a in it["answers"])
        per.append(dict(id=it["id"], pred=o, em=em, f1=f1))
    return dict(em=sum(p["em"] for p in per) / len(per), f1=sum(p["f1"] for p in per) / len(per), n=len(per)), per


@torch.no_grad()
def run_speed(model, tok, n=30, new=128):
    items = json.loads((DATA / "indommlu_eval.json").read_text(encoding="utf8"))[:n]
    prompts = [chat(tok, "Jelaskan secara singkat jawaban soal berikut untuk siswa.\n\n" + it["question"] + "\n" + "\n".join(it["options"]))
               for it in items]
    ttft, tps = [], []
    for p in prompts[:3]:  # pemanasan
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    for p in prompts:
        enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        sync(); t0 = time.time()
        model.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
        sync(); ttft.append(time.time() - t0)
        t0 = time.time()
        g = model.generate(**enc, max_new_tokens=new, min_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id)
        sync(); dt = time.time() - t0
        tps.append((g.shape[1] - enc.input_ids.shape[1]) / dt)
    med = lambda v: sorted(v)[len(v) // 2]
    return dict(ttft_s=med(ttft), tokens_per_s=med(tps), n=n, new_tokens=new), None


# ------------------------------------------------------------------ utama
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--prec", choices=["bf16", "int8", "nf4"], required=True)
    ap.add_argument("--tasks", nargs="+", default=["indommlu", "nusax", "tydiqa", "speed"])
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{a.model}_{a.prec}.json"
    res = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
    done = lambda t: t in res and not res[t].get("interrupted", False)
    todo = [t for t in a.tasks if not done(t)]
    if not todo:
        print(f"[lewati] {a.model} {a.prec}: semua tugas sudah selesai"); return
    a.tasks = todo
    tok, model, info = load(a.model, a.prec)
    res.update(model=a.model, prec=a.prec, **info, gpu=torch.cuda.get_device_name(0) if CUDA else "cpu",
               torch=torch.__version__)
    fn = dict(indommlu=run_indommlu, nusax=run_nusax, tydiqa=run_tydiqa, speed=lambda m, t, b: run_speed(m, t))
    for task in a.tasks:
        if task == "tydiqa" and not (DATA / "tydiqa_id_dev.json").exists():
            print("[lewati] data/tydiqa_id_dev.json belum ada; jalankan qid_data.py"); continue
        reset()
        with PowerMeter() as pm:
            metrics, per = fn[task](model, tok, a.batch)
        metrics.update(sec=round(pm.sec, 1), energy_wh=round(pm.wh, 3), mean_power_w=round(pm.mean_w, 1),
                       peak_vram_gb=round(peak(), 3), interrupted=pm.interrupted, max_gap_s=round(pm.max_gap, 1))
        if pm.interrupted:
            print(f"[peringatan] {task}: terdeteksi jeda {pm.max_gap:.0f} detik (sleep?). Tugas ini akan diulang pada run berikutnya.")
        res[task] = metrics
        if per is not None: res[task + "_items"] = per
        path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf8")
        show = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if not isinstance(v, dict)}
        print(f"[OK] {a.model} {a.prec} {task}: {show}")
    print("hasil ->", path)


if __name__ == "__main__":
    main()
