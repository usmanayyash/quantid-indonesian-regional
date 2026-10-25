"""
qid_app.py - purwarupa "Asisten Guru Offline": model bahasa kecil terkuantisasi yang berjalan di laptop tanpa internet.

Fitur: (1) tanya-jawab dari teks bacaan, (2) menjawab dan menjelaskan soal pilihan ganda, (3) meringkas materi.

Mode antarmuka (butuh: pip install gradio):
    python qid_app.py --model qwen3b --prec nf4
Mode uji kinerja di perangkat nyata (tanpa antarmuka, untuk naskah):
    python qid_app.py --model qwen3b --prec nf4 --bench
    python qid_app.py --model qwen1.5b --prec bf16 --bench --cpu     # laptop tanpa GPU
Hasil uji: results/app_{model}_{prec}_{gpu|cpu}.json
"""
import argparse, json, os, time
from pathlib import Path

import torch

import qid_eval as E

SYS = "Anda adalah asisten guru yang menjawab dalam bahasa Indonesia dengan singkat, jelas, dan benar."


def build(model_name, prec, cpu):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = os.environ.get("QID_LOCAL") or E.MODELS[model_name]
        tok = AutoTokenizer.from_pretrained(hf)
        model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32).eval()
        return tok, model
    tok, model, _ = E.load(model_name, prec)
    return tok, model


@torch.no_grad()
def ask(tok, model, user, max_new=200):
    msgs = [{"role": "user", "content": SYS + "\n\n" + user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    t0 = time.time()
    g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    dt = time.time() - t0
    n = g.shape[1] - enc.input_ids.shape[1]
    return tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip(), dt, n


def p_qa(teks, tanya): return f"Bacalah teks berikut lalu jawab pertanyaannya.\n\nTeks: {teks}\n\nPertanyaan: {tanya}"
def p_mcq(soal): return f"Jawablah soal pilihan ganda berikut. Sebutkan huruf jawaban lalu jelaskan alasannya dalam 2-3 kalimat.\n\n{soal}"
def p_sum(teks): return f"Ringkaslah materi berikut menjadi 3-5 poin penting untuk siswa.\n\n{teks}"


def scenarios():
    """20 skenario tetap untuk uji kinerja: 8 tanya-jawab bacaan, 8 soal pilihan ganda, 4 ringkasan."""
    mm = json.loads((E.DATA / "indommlu_eval.json").read_text(encoding="utf8"))
    sc = []
    ty = E.DATA / "tydiqa_id_dev.json"
    if ty.exists():
        for it in json.loads(ty.read_text(encoding="utf8"))[:8]:
            sc.append(("tanya-jawab", p_qa(it["context"][:2500], it["question"])))
    for it in mm[100:108]:
        sc.append(("soal", p_mcq(it["question"] + "\n" + "\n".join(it["options"]))))
    if ty.exists():
        for it in json.loads(ty.read_text(encoding="utf8"))[20:24]:
            sc.append(("ringkasan", p_sum(it["context"][:2500])))
    return sc


def bench(tok, model, tag, cpu=False):
    sc = scenarios()
    ask(tok, model, "Halo", 8)  # pemanasan
    E.reset()
    rows = []
    with E.PowerMeter() as pm:
        for kind, prompt in sc:
            out, dt, n = ask(tok, model, prompt)
            rows.append(dict(kind=kind, sec=round(dt, 2), new_tokens=n, tokens_per_s=round(n / dt, 2), answer=out[:400]))
            print(f"  {kind:12s} {dt:6.2f} s  {n:4d} token")
    lat = sorted(r["sec"] for r in rows)
    res = dict(tag=tag, n=len(rows), median_latency_s=lat[len(lat) // 2], p90_latency_s=lat[int(0.9 * (len(lat) - 1))],
               mean_tokens_per_s=sum(r["tokens_per_s"] for r in rows) / len(rows), total_sec=round(pm.sec, 1),
               energy_wh=None if cpu else round(pm.wh, 3), wh_per_answer=None if cpu else round(pm.wh / len(rows), 4),
               peak_vram_gb=None if cpu else round(E.peak(), 3), items=rows)
    E.OUT.mkdir(exist_ok=True)
    (E.OUT / f"app_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print({k: v for k, v in res.items() if k != "items"})


def ui(tok, model):
    import gradio as gr
    with gr.Blocks(title="Asisten Guru Offline") as app:
        gr.Markdown("## Asisten Guru Offline\nModel bahasa kecil yang berjalan di perangkat ini tanpa internet.")
        with gr.Tab("Tanya-jawab bacaan"):
            t, q, o = gr.Textbox(label="Teks bacaan", lines=8), gr.Textbox(label="Pertanyaan"), gr.Textbox(label="Jawaban", lines=6)
            gr.Button("Jawab").click(lambda a, b: "{}\n\n({:.1f} detik)".format(*ask(tok, model, p_qa(a, b))[:2]), [t, q], o)
        with gr.Tab("Soal pilihan ganda"):
            s, o2 = gr.Textbox(label="Soal dan pilihan jawaban", lines=8), gr.Textbox(label="Jawaban dan penjelasan", lines=6)
            gr.Button("Kerjakan").click(lambda a: "{}\n\n({:.1f} detik)".format(*ask(tok, model, p_mcq(a))[:2]), s, o2)
        with gr.Tab("Ringkas materi"):
            m, o3 = gr.Textbox(label="Materi", lines=10), gr.Textbox(label="Ringkasan", lines=8)
            gr.Button("Ringkas").click(lambda a: "{}\n\n({:.1f} detik)".format(*ask(tok, model, p_sum(a))[:2]), m, o3)
    app.launch(server_name="127.0.0.1", inbrowser=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(E.MODELS), default="qwen3b")
    ap.add_argument("--prec", choices=["bf16", "int8", "nf4"], default="nf4")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="jalankan di CPU (fp32) untuk simulasi laptop tanpa GPU")
    a = ap.parse_args()
    tok, model = build(a.model, a.prec, a.cpu)
    tag = f"{a.model}_{'fp32' if a.cpu else a.prec}_{'cpu' if a.cpu else 'gpu'}"
    bench(tok, model, tag, a.cpu) if a.bench else ui(tok, model)


if __name__ == "__main__":
    main()
