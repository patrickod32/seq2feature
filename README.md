# seq2feature

**A nano-probe that maps token sequences to sparse-autoencoder features** — distilling a Gemma-2-9B GemmaScope SAE into a **5.3 MB text-only probe** that runs on a CPU.

> An SAE reads a model's *internal activations* and tells you which of thousands of human-interpretable concepts are active. It needs the full model, a GPU, and the activation stream. **seq2feature asks:** does the model's *output text* carry enough of that signal to recover the same concepts — from text alone, in something tiny? Mostly, yes.

A 5.3 MB probe (int8) reproduces the SAE's concept readout on held-out text — **~19 of every 20 of its top concepts are ones the SAE actually fired** — beating pretrained-embedding and bag-of-words baselines, and failing *legibly* out of distribution. It reads only text: no SAE, no base model, no activations at inference.

This is a technical write-up and a thing to play with, not an anonymised paper. It shares the approach, the trained probe, an interactive UI, and enough data + notebooks to reproduce the training and the numbers.

---

## The claim, in numbers

All held-out (by-document split), SAE = ground truth. Two metrics — state which you're reading:

| | top-5 agreement | per-concept AUC |
|---|---|---|
| **seq2feature probe (5.3 MB)** | **0.956** | **0.90** |
| bag-of-words (ridge) | 0.84 | 0.75 |
| bge-small embedding, zero-shot | 0.23 | 0.53 |
| frequency-5 / random floor | 0.30 / 0.07 | — / 0.50 |
| SAE (the target) | 1.00 | 1.00 |

- **Top-5 agreement** = of the 5 concepts the probe ranks highest on a span, how many the SAE fired (SAE fires ~138 of 2,048 concepts/span, so floors are drawn in).
- **Per-concept AUC** = for one concept, can it rank the spans where the SAE fired above where it didn't.
- **Quantization:** int8 (5.3 MB) is lossless vs fp32; int4 (2.65 MB) costs ~2 points of top-5 agreement.
- **Out of distribution** (fresh SAE ground truth on medical + legal): probe AUC drops 0.90 → **0.75** but keeps its full lead (bge collapses to 0.55), and top-5 precision barely moves (0.94). It fails *legibly* — universal concepts (questions, negation) transfer; concepts tied to an absent domain (Python, LaTeX) drop out where that domain isn't present.
- **Robustness check:** the probe's skill is reading *text*, not exploiting concept co-occurrence — it beats an honest distribution-only null (0.90 vs 0.63) and even a co-occurrence *oracle* that sees the true firings of the other 137 concepts (0.90 vs 0.84).

---

## Quickstart

```bash
pip install torch numpy sentencepiece
python probe/infer.py "the water was cold and frozen solid by morning"
```

This loads the **shipped 5 MB int8 probe** (`mega_golf_probe_int8.pkl.gz`, self-contained) and scores any text into the SAE's concepts on a CPU:

```
[loaded int8 (5 MB, shipped)]
> the water was cold and frozen solid by morning
   +4.03  Cold, chilled, or frozen states and temperatures
   +3.52  Detects the concept of 'hard' in various contexts, often contrasted with 'soft'.
   ...
```

(Pass `--fp32` to load the 21 MB fp32 `.pth` instead — same outputs, minus the int8 rounding.)

### The interactive UI

The app also uses a frozen `bge-small` embedder (for the side-by-side baseline and some tabs), so it needs a couple more packages and a one-time ~33 MB model download:

```bash
pip install torch numpy sentencepiece sentence-transformers scikit-learn
HF_HUB_OFFLINE=0 python monitor_app/monitor_explorer.py   # first run downloads bge-small (~33 MB)
# subsequent runs: python monitor_app/monitor_explorer.py  (serves on http://localhost:7860)
```

A flat, information-dense suite: paste text and see the concepts the probe surfaces, per-token evidence, the architecture, a live CPU throughput benchmark, and an **agent-monitoring** demo — the cheap probe screens every message and escalates flagged spans to a Claude Code / sentry model (the tiered-oversight pattern). *Demonstrated, not evaluated as a safety system.* (The app also loads a small legacy 2B-SAE monitor for comparison; the seq2feature probe is `golf_final/`.)

**Explore** — the concepts the probe surfaces on a span, with per-token localization and an on-demand Claude review of the labels:

![The Explore tab: seq2feature's concept readout on a span, with token localization and a Claude review of the surfaced concepts.](monitor_app/assets/app-explore.png)

**Inside the probe** — the architecture end to end (own tokenizer → hashed unigram/bigram tables → 1.5M-param transformer → per-token concept head → span pool), with a live token trace:

![The Inside-the-probe tab: the seq2feature architecture with a per-token trace through the network.](monitor_app/assets/app-architecture.png)

**Firewall for agent traffic** — the tiered-oversight demo: two agents work a task, the 5.3 MB probe screens every message (~$0, ~12 ms, ~130× faster than the agents), and when a watched concept fires (here, API-key exfiltration) the message is **held** and a Claude sentry is summoned to judge it:

![The Agents firewall tab: the probe screens every agent message and escalates a flagged API-key-exfiltration span to a Claude sentry.](monitor_app/assets/app-firewall.png)

---

## Repo layout

```
golf_final/            the trained probe — weights, int8, tokenizer, ladder
  mega_golf_probe.pth        fp32 weights (+ cfg, captions, norm stats)
  mega_golf_probe_int8.pkl.gz  shipped 5.3 MB int8
  mega_tok.model             the probe's own 32k SentencePiece tokenizer
  ladder.json                the parameter-golf ladder (size vs accuracy per rung)
probe/infer.py         minimal standalone loader + scorer (CPU)
monitor_app/           the full UI (stdlib HTTP server, no framework)
relabel_bundle/        training data: SAE firings + spans + captions
  targets.npz                the SAE's per-(span,concept) firings (the free labels)
  spans.jsonl.gz             the text spans (+ teacher tokens for re-alignment)
  relabels.jsonl             the 2,048 concept captions
notebooks/
  00_generate_labels.ipynb   run Gemma+SAE, caption each concept from its firings
  01_train_probe.ipynb       train the probe + the parameter-golf ladder (Section 1)
  02_evaluate_probe.ipynb    baselines + accuracy vs the real SAE (Section 2)
  03_ood_test.ipynb          fresh SAE on medical + legal, legible-failure analysis
```

**Reproduce it:** `01_train_probe.ipynb` trains the probe from `relabel_bundle/` (Colab, ~minutes on a GPU); `02_evaluate_probe.ipynb` reproduces the baseline table and accuracy above on CPU, no GPU needed.

---

## Where the SAE and data come from

- **Base model:** `gemma-2-9b` (via transformer-lens)
- **SAE (teacher):** GemmaScope — `gemma-scope-9b-pt-res-canonical`, `layer_20/width_16k/canonical` (via sae-lens). 2,048 of the 16,384 concepts were kept (fired often enough to learn and validate honestly).
- **Training corpus** (mixed, ~free labels from the SAE): `HuggingFaceFW/fineweb`, `bigcode/the-stack-smol`, `open-thoughts/OpenThoughts-114k`, `open-r1/OpenR1-Math-220k`, `Anthropic/hh-rlhf`.
- **Out-of-distribution test** (never in training): `ccdv/pubmed-summarization`, `coastalcph/lex_glue` (ledgar).
- **Captions:** the SAE's stock labels were noisy, so each concept was re-captioned *from its own firing examples* by a cheap model (Gemini 2.5 Flash Lite, ~$0.80 total) and kept only if the caption could predict held-out firings.

To regenerate the SAE labels yourself, run `notebooks/00_generate_labels.ipynb` (needs an A100/80GB for the 9B teacher). Set your own `OPENROUTER_KEY` in the environment for the captioning step — no keys are committed.

---

## The idea (why this is a *competitive AI safety* project)

Safety onboarding usually has no loss function — "work on something impactful" is hard to score. This project picks a *measurable* target instead: distil an SAE into the smallest text-only probe that still reproduces its concept readout, and quantify it against dumb baselines with the floor drawn on every figure. It borrows the toolkit of the OpenAI parameter-golf challenge (hashed n-gram tables, factorized embeddings, calibrated quantization) and points it at interpretability.

The honest frame: **the probe is faithful to the *SAE*, not to ground truth.** It reproduces what the SAE does — inheriting the SAE's concepts and its errors — from text alone. That's the whole claim, measured, with its limits stated.

## Honest limits

- **Text-only ceiling.** It reads what surfaces in text; anything the model computes internally but never expresses is out of reach. It's a concept monitor, not a deception detector.
- **Coverage is corpus-bound.** Only 2,048 of the SAE's 16,384 concepts are covered, and only 17 are safety-related — a general concept lens, not a safety detector. Fixable with a targeted corpus, not an architecture change.
- **Not a full SAE replica.** It detects concepts per-concept (AUC 0.90); it does not reconstruct the SAE's full diffuse activation vector (whole-vector cosine 0.54), and isn't meant to.
- **One teacher, one layer.** Numbers are for this SAE at layer 20 of one 9B model. A different model's SAE is untested.

## Model & license

Weights and code are released for research and educational use. The base model and SAE are third-party (Google / GemmaScope) under their own licenses.
