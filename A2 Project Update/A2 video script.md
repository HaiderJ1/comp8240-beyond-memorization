# COMP8240 A2 Project Update – video script

Haider Jalal (61156183). Target 5:00 (allowed 4:30–5:30). About 680 spoken words at a relaxed pace.
Slides: `slides.html` (open in Edge, F11). Webcam in the bottom-right corner (the slides leave that space free).
Lines in *[brackets]* are what to do on screen, not to say.

---

## 1. Recap (0:00–0:30) · slide 1

Hi guys, I'm Haider, and this is my project update for COMP8240.

My project reproduces Staab et al., "Beyond Memorization", from ICLR 2024. The paper shows that large language models can work out private things about you, like your age, where you live or what you earn, just from ordinary Reddit comments. GPT-4 got 85.5% top-1 on real data.

I picked it because it's a real privacy risk, and because the headline number is hard to check: the real data was never released. So my question is: does it reproduce on the data that was released?

## 2. The paper's data (0:30–1:15) · slide 2

The paper has two datasets.

The main one is PersonalReddit: 520 real Reddit profiles, hand-labelled for eight attributes, with a hardness rating from 1 to 5 for each label. It was never released, for ethics reasons, so nobody can rerun the headline result.

What they did release is a synthetic set: 525 examples built from 40 made-up personas. Each example is a question and a bot-written reply of about 115 words, and each one targets a single attribute, like location or sex. On this set the paper reports GPT-4 at 73.7%. That's the number I'm trying to reproduce.

## 3. How the system works, and running it (1:15–2:45) · slide 3, then the terminal

So how does it actually work?

*[slide 3]* It's a pipeline. The comments go into a prompt that frames it as a guessing game. The model reasons about the clues and gives its top three guesses. Then the answer gets parsed and scored. Scoring has two layers. First, fixed rules: an exact match for things like sex, a fuzzy string match for locations, and plus or minus five years for age. Second, anything the rules can't decide goes to GPT-4 as a judge. That second layer matters, and I'll come back to it.

I used the authors' own prompt and scoring code, and swapped GPT-4 for an open model, Qwen 2.5 7B, so I can run it myself on a GPU.

*[switch to terminal, run:]* `type results\demo_23.txt` *(the saved output of `demo_one.py 23`; scroll from the prompt down to the answer)*

Here's one example. You can see the prompt, and the reply mentions "the Big Pants" building, the hutongs, and Nanluoguxiang. The model picks up those landmarks and guesses Beijing first, then Shanghai, then Guangzhou. The true answer is Beijing, so the rules mark it correct, and GPT-4 got the same answer in the paper.

*[run:]* `.venv\Scripts\python a2_baseline.py score`

And this runs the scoring over all 525 predictions and prints accuracy by hardness and by attribute.

## 4. Replication status (2:45–3:30) · slide 4

So, does it replicate? Partly, and the "partly" is the interesting bit.

The released file still has GPT-4's original answers and the authors' scores. When I recompute with the judge, I get 73.9%, and the hardness breakdown matches the paper exactly. But with the rules alone, the same answers only score 59.2%. So about 15 points of the headline come from GPT-4 marking GPT-4's answers. It's biggest for education, which goes from 5% to 56%, and occupation, which goes from 11% to 60%, because those are free text.

On the 373 examples the rules can score by themselves, Qwen gets 57.9% top-1 and GPT-4's answers get 77.7%. So an open 7B model is well behind, and the judge is a likely reason the number is hard to reproduce.

## 5. New datasets (3:30–4:30) · slide 5

For new data I'm doing two things.

The existing dataset is SynthPAI, from NeurIPS 2024. It has 7,823 synthetic comments from 300 personas, with labels that humans checked, so it's about fifteen times bigger than the released set. The challenge is that it's still synthetic and made by the same family of model as the judge.

The dataset I'm building is a perturbed copy of the 525 examples, with the same labels. One version is anonymised: explicit details like city names or ages become placeholders like LOCATION and AGE. The other is paraphrased: same facts, different words. The question is whether the model is really reasoning from evidence or just spotting surface cues. The challenge is that an LLM rewrite can drop a clue or leak one.

## 6. Construction progress (4:30–4:50) · slide 6

Both versions are built, 525 examples each, generated with Qwen from the released replies. I ran the same baseline on them: 39.9% top-1 anonymised and 50.1% paraphrased, against 57.9% on the original. Next I'll spot-check the rewrites by hand.

## 7. Wrap-up (4:50–5:00) · slide 7

So: the synthetic results reproduce, but only with GPT-4 as the judge. Next I'll test a current OpenAI model with and without the judge, then run SynthPAI. Thanks for watching.

---

## Before you record (about 20 minutes, on your PC)

1. Copy `N:\uni\COMP8240\A2 Project Update` to your PC, or work from N: directly.
2. For the `score` run you only need Python and one package:
   `py -3.11 -m venv .venv` → `.venv\Scripts\pip install python-Levenshtein` → `.venv\Scripts\python a2_baseline.py score` (takes seconds).
3. `demo_one.py 23` was run on the GPU box on 10 Oct; its output is `results/demo_23.txt`. Full steps: `A2 submission steps.md`.
4. Record in Zoom or OBS: share the screen, webcam in the bottom-right corner, export an MP4, and check the length is between 4:30 and 5:30.
