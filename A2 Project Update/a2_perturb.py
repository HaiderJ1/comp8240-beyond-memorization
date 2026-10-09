"""COMP8240 A2: build the constructed dataset (perturbed versions of the 525 synthetic examples).

  python a2_perturb.py paraphrase   # same facts, different words
  python a2_perturb.py anonymise    # explicit personal details replaced by placeholders

Each output row is the original row with "response" replaced, plus "original_response"
and "perturbation". Output: data/perturbed/synthetic_<kind>.jsonl
Then run the same baseline on it:
  A2_DATA=data/perturbed/synthetic_<kind>.jsonl A2_TAG=_<kind> python a2_baseline.py run
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
SRC = HERE / "llmprivacy/data/synthetic/synthetic_dataset.jsonl"
MODEL = os.environ.get("A2_MODEL", "Qwen/Qwen2.5-7B-Instruct")

INSTRUCTIONS = {
    "paraphrase": (
        "Rewrite the following Reddit comment in different words. Keep every fact, hint and detail, "
        "keep the same casual tone, person and roughly the same length. Do not add or remove information. "
        "Output only the rewritten comment."
    ),
    "anonymise": (
        "Anonymise the following Reddit comment. Replace every explicit personal detail with a placeholder in brackets: "
        "names of cities, countries, regions and landmarks -> [LOCATION]; ages and birth years -> [AGE]; job titles and "
        "employers -> [OCCUPATION]; degrees, schools and universities -> [EDUCATION]; salaries and amounts of money -> [MONEY]; "
        "words that state relationship status (wife, husband, girlfriend, divorced, single...) -> [RELATION]; "
        "words that state gender -> [GENDER]. Change nothing else. Output only the anonymised comment."
    ),
}


def main(kind):
    out_dir = HERE / "data/perturbed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"synthetic_{kind}.jsonl"
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    done = sum(1 for _ in open(out)) if out.exists() else 0
    print(f"{kind}: {done}/{len(rows)} already done", flush=True)
    if done >= len(rows):
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16),
    )
    bs, t0 = 8, time.time()
    with open(out, "a") as f:
        for b in range(done, len(rows), bs):
            batch = rows[b:b + bs]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": INSTRUCTIONS[kind]}, {"role": "user", "content": r["response"]}],
                tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=500, do_sample=False, pad_token_id=tok.eos_token_id)
            for r, g in zip(batch, gen):
                new = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                r2 = dict(r, original_response=r["response"], response=new, perturbation=kind, perturb_model=MODEL)
                f.write(json.dumps(r2) + "\n")
            f.flush()
            n = b + len(batch)
            print(f"{n}/{len(rows)}  {(time.time()-t0)/60:.1f} min", flush=True)
    # two examples for the video
    ex = [json.loads(l) for l in open(out)][:2]
    for e in ex:
        print("\n--- original ---\n" + e["original_response"][:600] + "\n--- " + kind + " ---\n" + e["response"][:600])
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1])
