from ..utils.common import read_wordlist
import re

def toxicity_lite(text: str, wordlist_path: str):
    wl = read_wordlist(wordlist_path)
    toks = re.findall(r"[\w']+", text.lower())
    hits = [t for t in toks if t in wl]
    return {"toxic_hits": len(hits), "terms": hits[:5]}
