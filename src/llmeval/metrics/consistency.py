import numpy as np

def self_consistency(samples):
    # numeric stability: calc pairwise similarity via Jaccard on tokens
    if not samples or len(samples)<2:
        return {"variance": None, "mean_jaccard": None}
    def jacc(a,b):
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb: return 0.0
        return len(sa & sb)/len(sa | sb)
    n = len(samples)
    sims = []
    for i in range(n):
        for j in range(i+1,n):
            sims.append(jacc(samples[i], samples[j]))
    return {"variance": float(np.var(sims)), "mean_jaccard": float(np.mean(sims))}
