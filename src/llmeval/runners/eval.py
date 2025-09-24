import argparse, json, os, pandas as pd, numpy as np
from tqdm import tqdm
from llmeval.utils.common import load_jsonl
from llmeval.providers import get_provider
from llmeval.metrics.relevance import relevance_scores
from llmeval.metrics.toxicity import toxicity_lite
from llmeval.metrics.bias import group_delta, weat_effect_size
from llmeval.metrics.consistency import self_consistency
from llmeval.judge.engine import JudgeEngine
from llmeval.report.html import render_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    import yaml
    from pathlib import Path

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        ap.error(f"Config file '{cfg_path}' not found.")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        ap.error(f"Config file '{cfg_path}' is empty.")
    # provider
    provider = get_provider(cfg.get('provider','openai'), **cfg)
    # rubric
    import json
    rubric = json.load(open(cfg['judge']['rubric'],'r'))
    engine = JudgeEngine(provider, rubric)

    # data
    ds = {r['id']: r for r in load_jsonl(cfg['dataset_path'])}
    gens = list(load_jsonl(cfg['generations_path']))

    # optional anchor calibration
    calib = {}
    anchors_path = cfg['judge'].get('anchors')
    if anchors_path and os.path.exists(anchors_path):
        anchors = list(load_jsonl(anchors_path))
        calib = engine.calibrate(anchors)

    out_rows = []
    models = set()

    # Precompute embeddings for references
    if cfg['metrics']['relevance'].get('use_embeddings', True):
        ref_texts = [ds[k]['reference'] for k in ds]
        ref_embs = provider.embed(ref_texts)
        ref_map = {k: ref_embs[i] for i,k in enumerate(ds.keys())}
    else:
        ref_map = {}

    for g in tqdm(gens, desc="Scoring"):
        _id = g['id']; output = g['output']; models.add(g.get('model','unknown'))
        item = ds.get(_id, {}); prompt = item.get('prompt',''); ref = item.get('reference','')
        # Relevance
        out_emb = None
        if cfg['metrics']['relevance'].get('use_embeddings', True):
            out_emb = provider.embed([output])[0]
        rel = relevance_scores(output, ref, out_emb, ref_map.get(_id), **cfg['metrics']['relevance'])
        # Toxicity
        tox = toxicity_lite(output, cfg['toxicity']['wordlist_path'])
        # Self-consistency (if multiple samples provided)
        sc = {}
        if 'samples' in g:
            sc = self_consistency([output]+g['samples'])
        # LLM-as-a-Judge
        judge_scores = {}
        if cfg['judge']['mode'] == 'pointwise':
            js = engine.score_pointwise(prompt, output)
            judge_scores = js.get('scores', {})
        row = {"id": _id, **rel, **tox, "judge_scores": judge_scores, **sc}
        out_rows.append(row)

    # Aggregates
    import pandas as pd, numpy as np
    df = pd.DataFrame(out_rows)
    # handle possibly missing columns
    def colmean(series):
        try:
            return float(series.mean())
        except Exception:
            return None
    agg = {
        "relevance_mean": colmean(df.get('relevance', pd.Series(dtype=float))),
        "semantic_mean": colmean(df.get('semantic', pd.Series(dtype=float))),
        "lex_f1_mean": colmean(df.get('lexical_f1', pd.Series(dtype=float))),
        "tox_hits_mean": colmean(df.get('toxic_hits', pd.Series(dtype=float))),
        "judge_rel_mean": colmean(df.get('judge_scores', pd.Series([{}]*len(df))).map(lambda x: x.get('relevance',np.nan)) if 'judge_scores' in df else pd.Series(dtype=float)),
        "anchor_acc": calib.get('anchor_accuracy')
    }

    os.makedirs(cfg['report']['out_dir'], exist_ok=True)
    df.to_json(os.path.join(cfg['report']['out_dir'], 'summary.json'), orient='records', indent=2)
    df.to_csv(os.path.join(cfg['report']['out_dir'], 'summary.csv'), index=False)
    render_report(out_rows, agg, models, os.path.join(cfg['report']['out_dir'], 'report.html'))
    print("Done. See reports in", cfg['report']['out_dir'])

if __name__ == '__main__':
    main()
