from ..utils.common import cosine, lexical_f1
import numpy as np

def relevance_scores(output: str, reference: str, out_emb=None, ref_emb=None, use_embeddings=True, use_lexical=True):
    scores = {}
    if use_embeddings and out_emb is not None and ref_emb is not None:
        scores['semantic'] = cosine(out_emb, ref_emb)
    if use_lexical:
        scores['lexical_f1'] = lexical_f1(output or '', reference or '')
    # aggregate
    vals = [v for v in scores.values() if isinstance(v,(int,float))]
    scores['relevance'] = float(np.mean(vals)) if vals else None
    return scores
