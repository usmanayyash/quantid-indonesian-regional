"""
qid_data.py - unduh TyDiQA-GoldP (dev) lalu ambil bagian bahasa Indonesia -> data/tydiqa_id_dev.json
(IndoMMLU dan NusaX sudah disertakan di folder data/.)

    python qid_data.py
"""
import json, urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json"
OUT = Path("data/tydiqa_id_dev.json")


def from_gcs():
    with urllib.request.urlopen(URL, timeout=120) as r:
        d = json.loads(r.read().decode("utf8"))
    items = []
    for art in d["data"]:
        for p in art["paragraphs"]:
            for q in p["qas"]:
                if q["id"].lower().startswith("indonesian"):
                    items.append(dict(id=q["id"], context=p["context"], question=q["question"],
                                      answers=sorted({a["text"] for a in q["answers"]})))
    return items


def from_hf():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="validation")
    return [dict(id=r["id"], context=r["context"], question=r["question"], answers=sorted(set(r["answers"]["text"])))
            for r in ds if r["id"].lower().startswith("indonesian")]


def main():
    try:
        items = from_gcs(); src = "Google Cloud Storage"
    except Exception as e:
        print("[info] unduhan langsung gagal (", e, "), mencoba Hugging Face ...")
        items = from_hf(); src = "Hugging Face"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False), encoding="utf8")
    print(f"[OK] {OUT}: {len(items)} pertanyaan bahasa Indonesia (sumber: {src})")


if __name__ == "__main__":
    main()
