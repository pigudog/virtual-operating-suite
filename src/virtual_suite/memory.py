"""Read-only retrieval from an optional, externally prepared experience store.
No experience entries are bundled with this package.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from .world import World, Day, LATE

K_PER_TYPE, TYPES = 2, ("principle", "pattern", "warning", "case")


def signature(w: World, day: Day, st: dict, t: float, fin: dict) -> list:
    rooms = sorted(st["assign"]); waiting = [g for r in rooms for g in st["assign"][r] if w.entry(g) > t and w.known(g, t)]
    over = {r: max(0.0, f - LATE) for r, f in fin.items()}
    depts = {}
    for g in waiting: depts[str(w.c["department"].iloc[g])] = depts.get(str(w.c["department"].iloc[g]), 0) + 1
    top = sorted(depts.items(), key=lambda x: -x[1])[:5]
    v = [len(rooms) / 12, len(waiting) / 40, sum(over.values()) / 600, sum(1 for o in over.values() if o > 0) / 6, sum(w.gmed[g] for g in waiting) / 3000]
    return v, [d for d, _ in top]


class ExperienceStore:
    def __init__(self, path: Path):
        self.path = Path(path); self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def retrieve(self, sig, depts, k=K_PER_TYPE):
        live = [e for e in self.entries if not e.get("retired")]
        if not live: return []
        v = np.asarray(sig, float); out = []
        for typ in TYPES:
            cand = []
            for e in live:
                if e.get("type") != typ: continue
                u = np.asarray(e["sig"], float); cos = float(v @ u / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9)); share = len(set(depts) & set(e.get("depts", []))) / max(len(depts), 1)
                cand.append((cos + 0.5 * share + 0.02 * np.tanh(e.get("utility", 0.0) / 30.0), e))
            cand.sort(key=lambda x: -x[0]); out += [e for _, e in cand[:k]]
        return out

    def render(self, entries: list[dict]) -> str:
        lab = {"principle": "Principle", "pattern": "Pattern that worked", "warning": "Warning", "case": "Similar past day"}
        return " | ".join(f"{lab[e['type']]}: {e['text']}" for e in entries)
