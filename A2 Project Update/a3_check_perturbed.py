"""COMP8240 A3: automatic check of the constructed (perturbed) datasets, no model needed.

For each of the 525 examples, does the ground-truth value still appear in the text?
  - original:   how often the answer is written out in the released reply (a "giveaway")
  - anonymise:  how often it survives anonymisation (a leak; should be near 0 for location/occupation)
  - paraphrase: how often it survives paraphrasing (should stay close to the original)
Also: placeholder counts, length change, and word-trigram overlap with the original.
Matching: case-insensitive; for "City, Country" values the city part alone counts; age = the number;
sex/income/relationship are skipped (their values are rarely written out literally).

  python a3_check_perturbed.py   -> results/perturbed_check.txt and results/perturbed_check.jsonl
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIG = HERE / "llmprivacy/data/synthetic/synthetic_dataset.jsonl"
KINDS = {"anonymise": HERE / "data/perturbed/synthetic_anonymise.jsonl",
         "paraphrase": HERE / "data/perturbed/synthetic_paraphrase.jsonl"}
OUT = HERE / "results"
CHECK = ["city_country", "birth_city_country", "occupation", "age", "education"]
PLACEHOLDERS = ["[LOCATION]", "[AGE]", "[OCCUPATION]", "[EDUCATION]", "[MONEY]", "[RELATION]", "[GENDER]"]


def needles(feature, gt):
    gt = str(gt).strip().lower()
    if feature == "age":
        return [gt]
    if feature in ("city_country", "birth_city_country"):
        return [gt.split(",")[0].strip()]
    return [gt]


def present(text, feature, gt):
    t = text.lower()
    for n in needles(feature, gt):
        if not n:
            continue
        if feature == "age":
            if re.search(rf"\b{re.escape(n)}\b", t):
                return True
        elif n in t:
            return True
    return False


def trigrams(s):
    w = re.findall(r"\w+", s.lower())
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def main():
    orig = [json.loads(l) for l in open(ORIG) if l.strip()]
    lines, rows = [], []
    base = defaultdict(lambda: [0, 0])
    for x in orig:
        f = x["feature"]
        if f in CHECK:
            base[f][1] += 1
            base[f][0] += present(x["response"], f, x["personality"][f])
    lines.append("Ground-truth value written out in the text (examples where it is, / examples of that attribute)")
    lines.append(f"{'attribute':20s} {'original':>14s} {'anonymised':>14s} {'paraphrased':>14s}")
    pert = {}
    for k, p in KINDS.items():
        pert[k] = [json.loads(l) for l in open(p) if l.strip()]
        assert len(pert[k]) == len(orig), (k, len(pert[k]))
    per = {k: defaultdict(lambda: [0, 0]) for k in KINDS}
    for i, x in enumerate(orig):
        f = x["feature"]
        rec = {"idx": i, "feature": f, "hardness": x["hardness"]}
        for k in KINDS:
            y = pert[k][i]
            if f in CHECK:
                hit_o = present(x["response"], f, x["personality"][f])
                hit_p = present(y["response"], f, x["personality"][f])
                per[k][f][1] += 1
                per[k][f][0] += hit_p
                rec[f"{k}_answer_in_text"] = hit_p
                rec["original_answer_in_text"] = hit_o
            to, tp = trigrams(x["response"]), trigrams(y["response"])
            rec[f"{k}_trigram_kept"] = round(len(to & tp) / max(1, len(to)), 3)
            rec[f"{k}_len_ratio"] = round(len(y["response"].split()) / max(1, len(x["response"].split())), 3)
            if k == "anonymise":
                rec["placeholders"] = sum(y["response"].count(ph) for ph in PLACEHOLDERS)
        rows.append(rec)
    tot = {"original": [0, 0], "anonymise": [0, 0], "paraphrase": [0, 0]}
    for f in CHECK:
        o, a, p = base[f], per["anonymise"][f], per["paraphrase"][f]
        for key, v in (("original", o), ("anonymise", a), ("paraphrase", p)):
            tot[key][0] += v[0]
            tot[key][1] += v[1]
        lines.append(f"{f:20s} {o[0]:>5d}/{o[1]:<3d} {100*o[0]/o[1]:4.0f}% {a[0]:>5d}/{a[1]:<3d} {100*a[0]/a[1]:4.0f}% {p[0]:>5d}/{p[1]:<3d} {100*p[0]/p[1]:4.0f}%")
    lines.append(f"{'ALL checked':20s} " + " ".join(f"{v[0]:>5d}/{v[1]:<3d} {100*v[0]/v[1]:4.0f}%" for v in tot.values()))
    lines.append("")
    for k in KINDS:
        kept = sorted(r[f"{k}_trigram_kept"] for r in rows)
        lr = sorted(r[f"{k}_len_ratio"] for r in rows)
        lines.append(f"{k}: word-trigrams kept from the original: mean {sum(kept)/len(kept):.2f}, median {kept[len(kept)//2]:.2f}; "
                     f"length ratio mean {sum(lr)/len(lr):.2f}, median {lr[len(lr)//2]:.2f}; "
                     f"unchanged (>=0.95 kept) {sum(v >= 0.95 for v in kept)}/{len(kept)}")
    ph = Counter()
    for y in pert["anonymise"]:
        for p in PLACEHOLDERS:
            ph[p] += y["response"].count(p)
    nz = sum(r["placeholders"] > 0 for r in rows)
    lines.append(f"anonymise: replies with at least one placeholder {nz}/{len(rows)}; placeholder counts {dict(ph)}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "perturbed_check.txt").write_text(txt + "\n")
    with open(OUT / "perturbed_check.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
