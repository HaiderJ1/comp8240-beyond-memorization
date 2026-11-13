# Does "Beyond Memorization" reproduce on its released data?

COMP8240 Applications of Data Science, Macquarie University, Session 2 2026. Haider Jalal.

This repository holds the code, new data and results for my COMP8240 project. The project replicates Staab et al., "Beyond Memorization: Violating Privacy via Inference with Large Language Models" (ICLR 2024), and tests it on new data. The final report is in `A3 Final Report/`.

## What is here

| Path | What it is |
|---|---|
| `A3 Final Report/` | The final report (PDF and LaTeX source, ACL template) |
| `A2 Project Update/llmprivacy/` | The authors' repository ([eth-sri/llmprivacy](https://github.com/eth-sri/llmprivacy)) as a git submodule, pinned at the commit I used. It holds the 525 released synthetic examples (`data/synthetic/synthetic_dataset.jsonl`) |
| `A2 Project Update/a2_baseline.py` | Replication: inspects the released data, recomputes the paper's numbers from the stored scores, runs an open-weight model with the authors' prompt, and scores it with the authors' rules (no GPT-4 judge) |
| `A2 Project Update/a2_perturb.py` | Builds the constructed dataset: anonymised and paraphrased versions of the 525 examples |
| `A2 Project Update/a3_check_perturbed.py` | Automatic check of the constructed data (is the answer still written in the text?), no model needed |
| `A2 Project Update/a3_synthpai.py` | Converts SynthPAI to the same task, then runs and scores the same model on it |
| `A2 Project Update/demo_one.py` | One example end to end, for a quick look |
| `A2 Project Update/data/perturbed/` | The new data: `synthetic_anonymise.jsonl` and `synthetic_paraphrase.jsonl`, with the same labels as the released set |
| `A2 Project Update/results/` | Every prediction and score behind the tables in the report |

SynthPAI itself is not copied here; it is on Hugging Face as `RobinSta/SynthPAI`. `a2_baseline.py synthpai` downloads it to `A2 Project Update/data/SynthPAI/`, and `a3_synthpai.py build` derives the author-level targets from it.

## Running it

Clone with the submodule:

```
git clone --recurse-submodules https://github.com/HaiderJ1/comp8240-beyond-memorization.git
cd "comp8240-beyond-memorization/A2 Project Update"
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

The model runs need an NVIDIA GPU with about 10 GB of memory (Qwen2.5-7B-Instruct in 4-bit NF4). The checks and the scoring of the stored predictions run on a CPU. All commands run from `A2 Project Update/`.

| Step | Command | Output |
|---|---|---|
| Recompute the paper's numbers from the released file | `python a2_baseline.py recorded` | `results/gpt4_recorded_scores.txt` |
| Open model on the 525 released examples | `python a2_baseline.py run`, then `python a2_baseline.py score` | `results/predictions_Qwen2.5-7B-Instruct.jsonl`, `results/scores_Qwen2.5-7B-Instruct.txt` |
| Build the perturbed data | `python a2_perturb.py anonymise` and `python a2_perturb.py paraphrase` | `data/perturbed/` |
| Open model on the perturbed data | `A2_DATA=data/perturbed/synthetic_anonymise.jsonl A2_TAG=_anonymise python a2_baseline.py run`, then the same with `score` (and again with `paraphrase`) | `results/*_anonymise.*`, `results/*_paraphrase.*` |
| Check the perturbed data | `python a3_check_perturbed.py` | `results/perturbed_check.txt` |
| SynthPAI | `python a2_baseline.py synthpai`, then `python a3_synthpai.py build`, then `A3_LIMIT=420 python a3_synthpai.py run` and `A3_LIMIT=420 python a3_synthpai.py score` | `results/*_synthpai.*` |

Runs resume where they stopped. `A2_MODEL` sets the model (default `Qwen/Qwen2.5-7B-Instruct`) and `A2_BS` the batch size.

## Notes

- Scoring uses the authors' rules only. The original pipeline also sends undecided cases to GPT-4 as a judge; the report measures how much of the headline result that judge accounts for.
- No credentials are needed or included. The authors' repository expects a `credentials.py` for OpenAI calls; it is not used here and is excluded by `.gitignore`.
- All the data is synthetic and was released for research by its authors. No real person's text was collected, and no attribute of a real person was inferred.
