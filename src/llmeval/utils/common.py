import json, os, math, numpy as np, pandas as pd, re
from typing import List, Dict

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def cosine(a, b):
    a = np.array(a); b = np.array(b)
    denom = (np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom else 0.0

def lexical_f1(a: str, b: str):
    # simple token F1
    ta = a.lower().split()
    tb = b.lower().split()
    common = set(ta) & set(tb)
    if not ta or not tb:
        return 0.0
    prec = len(common)/len(set(ta))
    rec = len(common)/len(set(tb))
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def read_wordlist(path):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set([w.strip().lower() for w in f if w.strip()])
