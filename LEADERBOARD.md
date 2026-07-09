# seq2feature leaderboard — *beat that*

A Competitive AI Safety target: **recover as much of a Gemma-2-9B SAE's concept readout as you can, from text alone, in as few bytes as possible.**

The score is **top-5 agreement** with the SAE on held-out text — of the 5 concepts a model ranks highest on a span, how many the SAE actually fired (SAE fires ~138 of 2,048 concepts per span, so the floors below are drawn in). Everything is measured on the same **by-document** held-out split (`crc32(doc) % 100 >= 70`), against the SAE's real firings in `relabel_bundle/targets.npz`. Reproduce or score a new entry with [`notebooks/02_evaluate_probe.ipynb`](notebooks/02_evaluate_probe.ipynb).

🎯 = the target (not a submission — it's the ground truth). Rank is over submissions between the floors and the target.

| rank | entry | top-5 agreement | size | reads activations? |
|:--:|---|:--:|:--:|:--:|
| 🎯 | **Gemma-2-9B + GemmaScope SAE** (teacher / target) | **1.000** | ≈18 GB | yes |
| **1** | **seq2feature — int8** *(current record)* | **0.956** | **5.3 MB** | **no** |
| — | seq2feature — int4 | 0.936 | 2.65 MB | no |
| 2 | bag-of-words (hashed n-grams → ridge) | 0.840 | ≈34 MB | no |
| — | *frequency-5 floor* (always name the 5 commonest) | 0.297 | — | — |
| 3 | bge-small embedding (zero-shot) | 0.230 | 33 MB | no |
| — | *random-5 floor* | 0.068 | — | — |

**Current record:** `seq2feature` (int8) — **0.956 top-5 agreement in 5.3 MB**, ~6× smaller than the strongest baseline (bag-of-words) and 0.116 above it, with no access to activations.

## How to enter

1. Train anything that maps **text → the 2,048 concepts** (weights, size, and architecture are yours to choose — the only rule is *no activations at inference*).
2. Score it with `notebooks/02_evaluate_probe.ipynb` on the fixed by-document split — top-5 agreement is the number.
3. Open a PR adding your row (entry, score, size, one-line method). A submission counts if the score reproduces from your code + weights.

## Why this is a leaderboard, not just a table

The point of Competitive AI Safety (see the write-up) is that a measurable target turns an open-ended problem into a game creative people can iterate on. This is one instance: an expensive safety signal (an SAE) with its cost/size/accuracy made into a score. Two axes are open to compete on — **higher top-5 agreement**, and **smaller size at fixed agreement**. Beat either and you've moved the frontier.

*Beat that.*
