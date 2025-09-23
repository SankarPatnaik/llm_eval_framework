from ..utils.common import cosine
import itertools, numpy as np, re

def counterfactual_swap(text: str, pairs):
    swapped = text
    for a,b in pairs:
        swapped = re.sub(rf'\b{re.escape(a)}\b', b, swapped, flags=re.I)
        swapped = re.sub(rf'\b{re.escape(b)}\b', a, swapped, flags=re.I)
    return swapped

def group_delta(outputs_by_group: dict, embedder):
    # compute mean embedding per group & pairwise distances
    group_emb = {g: np.mean(embedder(v), axis=0) for g,v in outputs_by_group.items() if v}
    deltas = {}
    keys = list(group_emb.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a,b = keys[i], keys[j]
            deltas[f"{a}|{b}"] = 1.0 - cosine(group_emb[a], group_emb[b])
    return deltas

def weat_effect_size(X, Y, A, B, embedder):
    # X,Y targets; A,B attributes
    ex = np.mean(embedder(X), axis=0); ey = np.mean(embedder(Y), axis=0)
    ea = np.mean(embedder(A), axis=0); eb = np.mean(embedder(B), axis=0)
    s_x = cosine(ex, ea) - cosine(ex, eb)
    s_y = cosine(ey, ea) - cosine(ey, eb)
    # pooled std (approximate with list of per-item sims if desired)
    std = np.std([s_x, s_y]) or 1.0
    return (s_x - s_y) / std
