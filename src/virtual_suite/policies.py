"""Deterministic reference policies using the shared action space and safety gate."""
from __future__ import annotations
import numpy as np
from .world import World, Day, LATE
from .trajectory import episode_world

def _remaining_load(w, cases, t):
    """Rule-heuristic surgery load from the known list and observed status only."""
    load = 0.0
    for g in cases:
        if not w.known(g, t): continue
        if w.entry(g) > t:
            load += w.gmed[g]
        elif w.clk["EXIT"][g] > t:  # the case is still present; its future exit VALUE is not read
            elapsed = max(t - float(w.clk["SSTART"][g]), 0.0)
            load += max(float(w.gmed[g]) - elapsed, 0.0)
    return load


from . import actions as AC


PEOPLE_TRIES = 3


def people_verdict(w, day, roles, st, act, t, candidate_forecast):
    """environment: the receiving room's nurse and anaesthetist and the case's surgeon (language personas, cast.py) are asked before the action is executed"""
    if roles is None: return True, "ok", ""
    return roles.cast(w, day).people(act, st, t, candidate_forecast)


def status_quo(w, day, rng, roles=None):
    w = episode_world(w, day)
    st = AC.init_state(w.day)
    st["outcome"] = w.finish(st)
    return st, []


def balance(w, day, rng, roles=None, t=AC.WAKES[0]):
    """the rule heuristic: once, at the first wake-up, move the longest remaining case of the most loaded room to the least loaded room"""
    w = episode_world(w, day); day = w.day
    st = AC.init_state(day); log = []
    w.advance(t, st)
    load = {r: _remaining_load(w, st["assign"][r], t) for r in st["assign"]}
    src = max(load, key=load.get); dst = min(load, key=load.get); rem = AC.remaining(w, st, src, t)
    if rem and src != dst:
        g = max(rem, key=lambda x: w.gmed[x]); act = {"type": "move", "case": g, "to": dst, "position": "end"}
        ok, code, why = AC.gate(w, day, st, act, t, roles=None)
        gain = None
        if ok:
            cand = AC.apply(w, st, act, t)
            base = w.simulate(day, st["assign"], t, n=12, rng=rng)
            alt = w.simulate(day, cand["assign"], t, n=12, rng=rng, detail=True)
            gain = AC.objective(base) - AC.objective(alt)
            if roles is not None:
                ok, code, why = roles.verdict(w, day, st, act, t, alt)
            if ok and gain < AC.GAIN_MIN:
                ok, why = False, f"gain below the minimum of {AC.GAIN_MIN:.0f} min"
            if ok:
                ok, code, why = people_verdict(w, day, roles, st, act, t, alt)
        log.append({"t": t, "proposals": [{"t": t, "action": act, "gate": code, "detail": why,
                    "executed": ok, "expected_gain_min": gain, "via": "execute"}],
                    "calls": [], "checks": [], "ended": "executed" if ok else "blocked"})
        if ok:
            w.commit(st, cand, t)
            st = cand
    st["outcome"] = w.finish(st)
    return st, log


def planner(w, day, rng, roles=None, n_dec=12):
    """the search over candidate actions (the agent's `plan` tool, run as an arm): at every wake-up, enumerate the candidate actions (moves to the
    earliest-finishing rooms, a swap, a reorder) of the most overloaded rooms, score each by OR-GPT (objective J = expected overtime), and execute the best one
    if it passes the gate and the role agents and improves J by >= GAIN_MIN; up to K_ACTIONS a day."""
    w = episode_world(w, day); day = w.day
    st = AC.init_state(day); log = []
    for t in AC.WAKES:
        if st["n_actions"] >= AC.K_ACTIONS: break
        w.advance(t, st)
        base = w.simulate(day, st["assign"], t, n=n_dec, rng=rng); fin = {r: float(v.mean()) for r, v in base["finish"].items()}
        if not fin or max(max(0.0, f - LATE) for f in fin.values()) <= 0: continue
        J0 = AC.objective(base); props = []; viable = []
        for act in AC.candidates(w, day, st, t, fin):
            ok, code, why = AC.gate(w, day, st, act, t, roles=None)
            if not ok: props.append({"t": t, "action": act, "gate": code, "executed": False, "expected_gain_min": None, "via": "search"}); continue
            cand = AC.apply(w, st, act, t); alt = w.simulate(day, cand["assign"], t, n=n_dec, rng=rng, detail=True)
            if roles is not None:
                ok, code, why = roles.verdict(w, day, st, act, t, alt)
                if not ok: props.append({"t": t, "action": act, "gate": code, "executed": False, "expected_gain_min": float(J0 - AC.objective(alt)), "via": "search"}); continue
            gain = J0 - AC.objective(alt); props.append({"t": t, "action": act, "gate": "ok", "executed": False, "expected_gain_min": float(gain), "via": "search"})
            if gain >= AC.GAIN_MIN: viable.append((gain, props[-1], cand, alt))
        ended = "hold"
        # Try the highest-gain feasible candidates with their original forecasts.
        viable.sort(key=lambda x: -x[0])
        for gain, p, candidate, alt in viable[:PEOPLE_TRIES]:
            ok, code, why = people_verdict(w, day, roles, st, p["action"], t, alt)
            if not ok: p["gate"] = code; p["detail"] = why; p["via"] = "execute"; continue
            w.commit(st, candidate, t)
            st = candidate; p["executed"] = True; p["via"] = "execute"; ended = "executed"; break
        log.append({"t": t, "proposals": props, "calls": [], "checks": [], "ended": ended})
    st["outcome"] = w.finish(st)
    return st, log



POLICIES = {"status_quo": status_quo, "balance": balance, "planner": planner}
