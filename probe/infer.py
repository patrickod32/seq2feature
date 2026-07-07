"""
seq2feature — minimal standalone inference.

Loads the 5 MB text-only probe (golf_final/) and scores arbitrary text into the
2,048 sparse-autoencoder concepts it was distilled to predict. No SAE, no base
model, CPU-only.

    python probe/infer.py "the water was cold and frozen solid by morning"
    python probe/infer.py            # runs a few built-in examples

The probe maps a token *sequence* to a *feature* vector (seq2feature): for each
concept it emits a normalised score; we print the top-k by score.
"""
import os, sys, json, gzip, numpy as np, torch, torch.nn as nn
import sentencepiece as spm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLF = os.path.join(ROOT, "golf_final")
CAPS_PATH = os.path.join(ROOT, "relabel_bundle", "relabels.jsonl")


def load():
    ck = torch.load(os.path.join(GOLF, "mega_golf_probe.pth"), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    V = [int(x) for x in ck["V"]]
    N_V = len(V)
    mu = np.asarray(ck["mu"], np.float32); sd = np.asarray(ck["sd"], np.float32)
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

    model = Probe().eval(); model.load_state_dict(ck["state_dict"])
    sp = spm.SentencePieceProcessor(model_file=os.path.join(GOLF, "mega_tok.model"))
    caps = {}
    if os.path.exists(CAPS_PATH):
        for line in open(CAPS_PATH):
            r = json.loads(line); caps[int(r["latent"])] = r.get("caption", "")
    captions = [caps.get(v, f"latent {v}") for v in V]
    return model, sp, mu, sd, captions, BIGH, MAXLEN


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
    bits = load()
    examples = sys.argv[1:] or [
        "the water was cold and frozen solid by morning",
        "def __init__(self, config): self.model = load(config)",
        "I'm sorry, but I can't help with that request.",
        "Call 555-0142 or SSN 078-05-1120 today",
    ]
    for text in examples:
        print(f"\n> {text[:70]}")
        for cap, score in top_concepts(text, *bits):
            print(f"   {score:+.2f}  {cap}")
