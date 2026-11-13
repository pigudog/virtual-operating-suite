"""actions.py: the action space, state and safety gate shared by every agent of the virtual operating suite.

State  = {"assign": {room: [case idx, ...] in order}, "n_actions": int}
Actions (JSON, the same for the agent, its `plan` tool and the rule heuristic); no case is cancelled or postponed to another day:
  {"type": "move",    "case": id, "to": room, "position": "end" | k}   move a not-yet-started case to another room (k-th among its remaining cases)
  {"type": "swap",    "case_a": id, "case_b": id}                      exchange two not-yet-started cases (rooms and positions)
  {"type": "reorder", "case": id, "position": k}                       move a not-yet-started case within its own room
Gate (every executed action must satisfy all of these; counted, never bypassed):
  G1 no referenced case has started     G2 a receiving room has hosted the case's procedure group (capability)
  G3 no overlapping sampled surgery intervals for the same surgeon across different rooms
  G4 emergencies are never moved, swapped or pushed later     G5 patients aged >= AGE_PROTECT are never moved, swapped or pushed later
  G6 at most K_ACTIONS executed actions a day     G7 no case is postponed to another day
Role agents (roles.py) add two further verdicts: S1 the surgeon refuses (outside their availability window) and R1 the receiving room refuses. The people of the
suite (cast.py) add three binding ones: N1 the receiving room's nurse, A1 its anaesthetist and S2 the case's surgeon.
Objective (the acceptance rule of every agent): J = expected suite overtime after 17:30 (min); an action is executed only
if J falls by >= GAIN_MIN under OR-GPT (n_dec continuations). Simulation throughout.
"""
from __future__ import annotations
import numpy as np
from .world import World, Day, LATE

K_ACTIONS, GAIN_MIN, AGE_PROTECT = 3, 10.0, 75.0
WAKES = [11 * 60, 12 * 60, 13 * 60, 14 * 60, 15 * 60, 16 * 60]
FORBIDDEN = {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}


def init_state(day: Day) -> dict:
    return {"assign": {r: list(v) for r, v in day.rooms.items()}, "n_actions": 0}


def copy_state(st: dict) -> dict:
    return {"assign": {r: list(v) for r, v in st["assign"].items()}, "n_actions": st["n_actions"]}


def room_of(st: dict, g: int):
    return next((r for r, v in st["assign"].items() if g in v), None)


def remaining(w: World, st: dict, room: int, t: float) -> list:
    return [g for g in st["assign"][room] if w.entry(g) > t and w.known(g, t)]


def _insert(w, st, room, g, position, t):
    lst = st["assign"][room]; started = [h for h in lst if w.entry(h) <= t]; rem = [h for h in lst if w.entry(h) > t]
    k = len(rem) if position in (None, "end", "") else int(max(0, min(int(position), len(rem))))
    rem.insert(k, g); st["assign"][room] = started + rem


def apply(w: World, st: dict, act: dict, t: float) -> dict:
    """apply an action without checks (the gate decides first); returns a new state"""
    s = copy_state(st); typ = act.get("type")
    if typ == "move":
        g = act["case"]; src = room_of(s, g); s["assign"][src].remove(g); _insert(w, s, act["to"], g, act.get("position", "end"), t)
    elif typ == "swap":
        a, b = act["case_a"], act["case_b"]; ra, rb = room_of(s, a), room_of(s, b); ia, ib = s["assign"][ra].index(a), s["assign"][rb].index(b)
        s["assign"][ra][ia], s["assign"][rb][ib] = b, a
    elif typ == "reorder":
        g = act["case"]; r = room_of(s, g); s["assign"][r].remove(g); _insert(w, s, r, g, act.get("position", "end"), t)
    else:
        raise ValueError("Only move, swap and reorder actions are supported")
    s["n_actions"] += 1
    return s


def protected_order_verdict(w, before, after, t):
    """Reject direct relocation or any newly preceding case for a waiting protected patient.

    This is a conservative list-order constraint, not a guarantee of a clinical
    start time. Removing predecessors or advancing a protected patient is safe;
    inserting/reordering other work ahead of them must also be checked.
    """
    for r, cases in before["assign"].items():
        waiting = [g for g in cases if w.entry(g) > t and w.known(g, t)]
        for i, g in enumerate(waiting):
            emergency = w.is_emergency(g)
            elderly = np.isfinite(w.age[g]) and w.age[g] >= AGE_PROTECT
            if not (emergency or elderly): continue
            code = "G4" if emergency else "G5"
            r2 = room_of(after, g)
            if r2 != r: return False, code, f"protected case {g} would leave its room or today's list"
            new = [h for h in after["assign"][r] if w.entry(h) > t and w.known(h, t)]
            predecessors = new[:new.index(g)]
            # Every new predecessor must remain in the same order within the old prefix.
            allowed = iter(waiting[:i])
            if not all(any(h == old for old in allowed) for h in predecessors):
                return False, code, f"other work would be inserted or reordered ahead of protected case {g}"
    return True, "ok", "protected waiting order preserved"


def gate(w: World, day: Day, st: dict, act: dict, t: float, roles=None, fc_after=None) -> tuple[bool, str, str]:
    """(ok, code, detail). Codes: ok | G0 invalid | G1..G7 | S1 surgeon window | R1 room refuses."""
    typ = act.get("type"); A = st["assign"]
    if typ == "defer": return False, "G7", "no case may be postponed to another day"
    try:
        if typ == "move": cases, dst = [int(act["case"])], int(act["to"])
        elif typ == "swap": cases, dst = [int(act["case_a"]), int(act["case_b"])], None
        elif typ == "reorder": cases, dst = [int(act["case"])], None
        else: return False, "G0", "unknown action type"
    except Exception: return False, "G0", "malformed action"
    for g in cases:
        if room_of(st, g) is None: return False, "G0", f"case {g} is not on today's list"
        if not w.known(g, t): return False, "G0", f"case {g} has not arrived"
    if typ == "move" and (dst not in A or dst == room_of(st, cases[0])): return False, "G0", "destination is not another room of the suite"
    if typ == "swap" and cases[0] == cases[1]: return False, "G0", "swap needs two different cases"
    if st["n_actions"] >= K_ACTIONS: return False, "G6", f"the daily cap of {K_ACTIONS} actions is reached"
    for g in cases:
        if w.entry(g) <= t: return False, "G1", f"case {g} has already started"
    for g in cases:
        if w.is_emergency(g): return False, "G4", f"case {g} is an emergency"
    if typ in ("move", "swap"):
        for g in cases:
            if np.isfinite(w.age[g]) and w.age[g] >= AGE_PROTECT: return False, "G5", f"patient of case {g} is aged {w.age[g]:.0f}"
    try:
        candidate = apply(w, st, act, t)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, "G0", "malformed action or position"
    ok, code, why = protected_order_verdict(w, st, candidate, t)
    if not ok: return False, code, why
    # Capability of the receiving room(s).
    moves = [(cases[0], dst)] if typ == "move" else ([(cases[0], room_of(st, cases[1])), (cases[1], room_of(st, cases[0]))] if typ == "swap" else [])
    for g, r in moves:
        if r == room_of(st, g): continue
        if str(w.c["procedure_group"].iloc[g]) not in w.room_cap.get(r, set()): return False, "G2", f"room {r} has never hosted {w.c['procedure_group'].iloc[g]}"
    ok, code, why = surgeon_overlap_verdict(w, day, candidate, t)
    if not ok: return False, code, why
    if roles is not None:
        ok, code, why = roles.verdict(w, day, st, act, t, fc_after)
        if not ok: return False, code, why
    return True, "ok", "authorised"


def surgeon_overlap_verdict(w, day, candidate, t):
    """Check each sampled surgery interval in the complete proposed schedule.

    Fixed 20 continuations and a local seed make the gate reproducible without
    consuming the caller's random stream. Only known cases are considered;
    future times come from simulation, not recorded future outcomes. Intervals
    are [surgery start, surgery end), so touching endpoints are allowed. This
    checks each continuation because averaging different surgery orders can
    create apparent overlaps even when every individual trajectory is feasible.
    """
    surgeons = {}
    for room, cases in candidate["assign"].items():
        for g in cases:
            if not w.known(g, t): continue
            surgeon = str(w.surgeon[g]).strip()
            if surgeon.lower() in {"", "nan", "none", "<na>", "missing", "unknown"}:
                return False, "G3", f"case {g} has no surgeon identifier to check"
            surgeons.setdefault(surgeon, []).append((room, g))
    shared = [cases for cases in surgeons.values() if len({r for r, _ in cases}) > 1]
    if not shared:
        return True, "ok", "no surgeon is assigned to multiple rooms"
    try:
        events = w.simulate(day, candidate["assign"], t, n=20,
                            rng=np.random.default_rng(0), detail=True).get("event_samples", {})
    except ValueError as exc:
        return False, "G3", f"cannot construct a valid surgeon schedule: {exc}"
    for cases in shared:
        intervals = []
        for room, g in cases:
            try:
                clocks = np.asarray(events[room][g], dtype=float)
                if (clocks.shape != (20, 4) or not np.isfinite(clocks).all()
                        or np.any(np.diff(clocks, axis=1) < 0)
                        or np.any(clocks[:, 2] <= clocks[:, 1])):
                    raise ValueError("invalid event times")
            except (KeyError, TypeError, ValueError):
                return False, "G3", f"case {g} has no valid sampled surgery intervals"
            start, end = clocks[:, 1], clocks[:, 2]
            for other_room, other_case, other_start, other_end in intervals:
                if room != other_room and np.any(np.maximum(np.maximum(start, other_start), t) < np.minimum(end, other_end)):
                    return False, "G3", f"cases {other_case} and {g} have overlapping sampled surgery intervals in different rooms"
            intervals.append((room, g, start, end))
    return True, "ok", "sampled surgery intervals do not overlap across rooms"


def audit_executed(w, day, log, *, seed=0):
    """Replay executed actions through the deterministic gate; do not assume zero breaches.

    Supply the original World and the same environment seed used in the run. Does not call role/persona services. Covers G0--G7 only.
    """
    if getattr(w, "is_episode", False):
        raise ValueError("Audit from the original World with the run's episode configuration")
    trajectory = isinstance(w, World)
    if trajectory:
        w = w.start_episode(day, seed=seed)
        day = w.day
    st = init_state(day); breaches = {}; checked = 0
    for episode in log:
        for p in episode.get("proposals", []):
            if not p.get("executed"): continue
            checked += 1
            t = p.get("t", episode.get("t"))
            try:
                if trajectory: w.advance(t, st)
                ok, code, _ = gate(w, day, st, p["action"], t)
                if not ok:
                    breaches[code] = breaches.get(code, 0) + 1
                    continue
                candidate = apply(w, st, p["action"], t)
                if trajectory: w.commit(st, candidate, t)
                st = candidate
            except (KeyError, TypeError, ValueError, OverflowError):
                breaches["G0"] = breaches.get("G0", 0) + 1
    return {"checked": checked, "breaches": sum(breaches.values()), "by_code": breaches}


def objective(out: dict) -> float:
    return float(out["overtime_min"].mean())


def candidates(w: World, day: Day, st: dict, t: float, fin: dict, n_rooms=2, n_cases=2, n_dst=3) -> list[dict]:
    """search space of the `plan` tool at one wake-up: for the n_rooms rooms with the largest expected overtime, each of their last n_cases remaining cases
    may be moved to the n_dst earliest-finishing other rooms (end), swapped with the shortest remaining case of the earliest-finishing room,
    or moved up within its room."""
    over = {r: max(0.0, f - LATE) for r, f in fin.items()}; srcs = [r for r in sorted(over, key=over.get, reverse=True) if over[r] > 0][:n_rooms]
    out = []
    for src in srcs:
        rem = remaining(w, st, src, t)
        for g in rem[-n_cases:]:
            dsts = sorted((r for r in fin if r != src), key=fin.get)[:n_dst]
            for d in dsts: out.append({"type": "move", "case": g, "to": d, "position": "end"})
            if dsts:
                rb = remaining(w, st, dsts[0], t)
                if rb: out.append({"type": "swap", "case_a": g, "case_b": min(rb, key=lambda x: w.gmed[x])})
            if len(rem) > 1 and g != rem[0]: out.append({"type": "reorder", "case": g, "position": 0})
    return out


def describe(w: World, act: dict) -> str:
    cid = lambda g: str(w.c["case_id"].iloc[int(g)])
    typ = act.get("type")
    if typ == "move": return f"move case {cid(act['case'])} to room {act['to']} ({act.get('position', 'end')})"
    if typ == "swap": return f"swap cases {cid(act['case_a'])} and {cid(act['case_b'])}"
    if typ == "reorder": return f"reorder case {cid(act['case'])} to position {act.get('position', 'end')}"
    return str(act)
