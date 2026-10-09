"""COMP8240 A2 live demo for the video: one synthetic profile, the paper's prompt, one model answer, scored.

  python demo_one.py            # example 23 (the same one the authors' demo.py uses)
  python demo_one.py 140        # any index 0-524

Uses the same prompt, parser and scoring rules as a2_baseline.py.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import a2_baseline as a

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 23
rows = a.rows()
x = rows[idx]
prompts, _ = a.build_prompts()
p = prompts[idx]

print("=" * 70)
print(f"Example {idx}: feature = {x['feature']}, hardness = {x['hardness']}")
print("=" * 70)
print("SYSTEM:", p["system"])
print("\nUSER PROMPT:\n" + p["user"])

tok = AutoTokenizer.from_pretrained(a.MODEL)
kw = {"device_map": "auto"}
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
except ImportError:
    kw["torch_dtype"] = torch.float16
print(f"\nLoading {a.MODEL} ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(a.MODEL, **kw)
text = tok.apply_chat_template([{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}],
                               tokenize=False, add_generation_prompt=True)
enc = tok(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=700, do_sample=False, pad_token_id=tok.eos_token_id)
answer = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

pii = a.FEATURE_MAP[x["feature"]]
gt = x["personality"][x["feature"]]
guesses = a.parse_guesses(answer, pii)
s = a.score_one(pii, gt, guesses)
print("\n" + "=" * 70 + "\nMODEL ANSWER:\n" + answer)
print("=" * 70)
print(f"Parsed guesses : {guesses}")
print(f"Ground truth   : {gt}")
print(f"Rule-based     : " + ("needs the GPT-4 judge (no rule for this value)" if s is None else f"top-1 {'correct' if s[0] else 'wrong'}, top-3 {'correct' if s[1] else 'wrong'}"))
print(f"GPT-4 in paper : {a.parse_guesses(x['guess'], pii)}  (authors' judge-aided score: top-1 {x['guess_correctness']['model_aided_eval'][0]})")
