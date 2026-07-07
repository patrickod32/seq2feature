"""
seq2feature — minimal standalone inference.

Loads the shipped 5 MB int8 probe (golf_final/mega_golf_probe_int8.pkl.gz) and
scores arbitrary text into the 2,048 sparse-autoencoder concepts it was distilled
to predict. No SAE, no base model, CPU-only. The int8 pickle is self-contained
(weights + config + captions + norm stats) — nothing else is needed.

    python probe/infer.py "the water was cold and frozen solid by morning"
    python probe/infer.py            # runs a few built-in examples
    python probe/infer.py --fp32 ... # use the 21 MB fp32 .pth instead

The probe maps a token *sequence* to a *feature* vector (seq2feature): for each
concept it emits a normalised score; we print the top-k by score.
"""
import os, sys, gzip, pickle, numpy as np, torch, torch.nn as nn
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLF = os.path.join(ROOT, "golf_final")
INT8 = os.path.join(GOLF, "mega_golf_probe_int8.pkl.gz")
FP32 = os.path.join(GOLF, "mega_golf_probe.pth")


def build(cfg, N_V):
    UNI, BIGH, CDIM, FDIM = cfg["UNI"], cfg["BIG_H"], cfg["CDIM"], cfg["FDIM"]
    NH, NL, MAXLEN = cfg["NHEAD"], cfg["NLAYER"], cfg["MAXLEN"]

    class Probe(nn.Module):
        def __init__(s):
            super().__init__(); d, e = CDIM, FDIM
            s.uni = nn.Embedding(UNI, e); s.big = nn.Embedding(BIGH, e)
            s.proj = nn.Linear(e, d, bias=False); s.pos = nn.Embedding(MAXLEN, d)
            s.enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, NH, 4 * d, batch_first=True, dropout=0.1),
                NL, enable_nested_tensor=False)
            s.head = nn.Linear(d, N_V)

        def forward(s, u, b, m):
            x = s.proj(s.uni(u) + s.big(b)) + s.pos(torch.arange(u.shape[1])[None])
            t = s.head(s.enc(x, src_key_padding_mask=~m))
            mk = t.masked_fill(~m.unsqueeze(-1), -1e30)
            return torch.logsumexp(mk, 1) - torch.log(m.sum(1, keepdim=True).float())

    return Probe(), MAXLEN, BIGH


def _dequant(entry):
    """int8 pickle stores each tensor as ('i8', int8[N,D], fp16 row-scale[N]) or ('f16', fp16, None)."""
    tag = str(entry[0])
    if tag == "i8":
        w = entry[1].astype(np.float32) * np.asarray(entry[2], np.float32)[:, None]
    else:  # f16 — biases / norms kept in half precision
        w = np.asarray(entry[1], np.float32)
    return torch.tensor(w)


def load_int8():
    d = pickle.load(gzip.open(INT8, "rb"))
    meta, q = d["meta"], d["q"]
    N_V = len(meta["V"])
    model, MAXLEN, BIGH = build(meta["cfg"], N_V)
    model.load_state_dict({k: _dequant(v) for k, v in q.items()})
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=os.path.join(GOLF, "mega_tok.model"))
    return model, sp, np.asarray(meta["mu"], np.float32), np.asarray(meta["sd"], np.float32), \
        list(meta["captions"]), BIGH, MAXLEN


def load_fp32():
    ck = torch.load(FP32, map_location="cpu", weights_only=False)
    N_V = len(ck["V"])
    model, MAXLEN, BIGH = build(ck["cfg"], N_V)
    model.load_state_dict(ck["state_dict"]); model.eval()
    sp = spm.SentencePieceProcessor(model_file=os.path.join(GOLF, "mega_tok.model"))
    caps = ck.get("captions") or [f"latent {v}" for v in ck["V"]]
    return model, sp, np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32), \
        list(caps), BIGH, MAXLEN


def bigram(ids, BIGH):
    return np.concatenate([[0], (36313 * ids[1:] ^ 27191 * ids[:-1]) % BIGH]).astype(np.int64)


@torch.no_grad()
def top_concepts(text, model, sp, mu, sd, captions, BIGH, MAXLEN, k=8):
    ids = sp.encode(text)[:MAXLEN] or [0]
    u = torch.tensor(ids)[None]
    b = torch.tensor(bigram(np.asarray(ids), BIGH))[None]
    m = torch.ones_like(u, dtype=torch.bool)
    z = (model.forward(u, b, m)[0].numpy() - mu) / sd
    return [(captions[c][:80], float(z[c])) for c in np.argsort(-z)[:k]]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--fp32"]
    bits = load_fp32() if "--fp32" in sys.argv else load_int8()
    which = "fp32 (21 MB)" if "--fp32" in sys.argv else "int8 (5 MB, shipped)"
    print(f"[loaded {which}]")
    examples = args or [
        "the water was cold and frozen solid by morning",
        "def __init__(self, config): self.model = load(config)",
        "I'm sorry, but I can't help with that request.",
        "Call 555-0142 or SSN 078-05-1120 today",
    ]
    for text in examples:
        print(f"\n> {text[:70]}")
        for cap, score in top_concepts(text, *bits):
            print(f"   {score:+.2f}  {cap}")
