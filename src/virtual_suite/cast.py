"""Generic simulated role dialogue and binding action verdicts.
Facts are built from the supplied simulation state; no patient records or learned
experience entries are bundled with this package.
"""
from __future__ import annotations
import json, re
import numpy as np
from .world import World, Day, LATE
from .roles import Roles
from .llm_client import chat

THINK_OFF = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
SYS = ("You are the {role} in a hospital operating suite, in a simulation used to test a coordinator. Answer the coordinator in one or two plain sentences, "
       "as this person would, using ONLY the facts given. Then give your stance. Reply as JSON: {{\"answer\": \"...\", \"stance\": \"agree|refuse|neutral\", \"reason\": \"...\"}}")


def hhmm(m):
    return "unknown" if m is None or not np.isfinite(m) else f"{int(m // 60):02d}:{int(m % 60):02d}"


def _int(x):
    """a room number from the model: an empty or malformed value is answered with an error message, never fatal"""
    try: return int(x)
    except (TypeError, ValueError): return None


class Cast:
    def __init__(self, w: World, day: Day, roles: Roles, log: list, scenario_info: dict | None = None):
        self.w, self.day, self.roles, self.log, self.info = w, day, roles, log, scenario_info
        self.calls = 0; self._detail = {}

    # ---- facts (data-grounded) ----
    def _with_events(self, st, t, sim):
        """the people reason on the expected times of single cases (OR-GPT, mean over continuations): the simulation with case-level events, once per state of the day"""
        if sim.get("events"): return sim
        key = (round(float(t), 1), tuple((r, tuple(v)) for r, v in sorted(st["assign"].items())))
        if key not in self._detail: self._detail = {key: self.w.simulate(self.day, st["assign"], t, n=12, rng=np.random.default_rng((abs(hash(key)) + 5) % 2**32), detail=True)}
        return self._detail[key]

    def _worklist(self, st, t, ev, mine, *, surgery=False):
        """the cases selected by `mine` that are not finished at time t, with the times OR-GPT expects"""
        w = self.w; out = []
        for rr, gs in sorted(st["assign"].items()):
            for g in gs:
                if w.arrival.get(g, -1) > t or not mine(g): continue
                start_key, end_key = ("SSTART", "SEND") if surgery else ("ENTRY", "EXIT")
                started = float(w.clk[start_key][g]) <= t
                if started and float(w.clk[end_key][g]) <= t: continue
                e = ev.get(rr, {}).get(int(g))
                if e is not None: out.append({"room": rr, "g": int(g), "in": float(e[1 if surgery else 0]), "out": float(e[2 if surgery else 3]), "running": started})
        return out

    def _render(self, items):
        w = self.w
        return "; ".join(f"room {x['room']}, case {str(w.c['case_id'].iloc[x['g']])[:8]} ({w.c['procedure_group'].iloc[x['g']]}), {hhmm(x['in'])}-{hhmm(x['out'])}" + (" (running now)" if x["running"] else "") for x in items) or "none"

    def _case_in(self, st, text):
        """the case the agent's question names, when it did not pass it as an argument"""
        m = re.search(r"\bC[0-9a-f]{7,}\b", str(text))
        if not m: return None
        return next((int(g) for gs in st["assign"].values() for g in gs if str(self.w.c["case_id"].iloc[g]).startswith(m.group(0)[:8])), None)

    @staticmethod
    def _window(ev, rf, g, r, *, surgery=False):
        """Read a case's actual forecast in the supplied schedule; never infer an insertion."""
        e = ev[r][int(g)]
        return float(e[1 if surgery else 0]), float(e[2 if surgery else 3]), rf[r]["finish"]

    def _room_facts(self, st, t, sim):
        w = self.w; out = {}
        for r in sorted(st["assign"]):
            cases = [g for g in st["assign"][r] if w.arrival.get(g, -1) <= t]; done = [g for g in cases if float(w.clk["EXIT"][g]) <= t and w.entry(g) <= t]
            running = [g for g in cases if w.entry(g) <= t and float(w.clk["EXIT"][g]) > t]; waiting = [g for g in cases if w.entry(g) > t]
            f = sim["finish"].get(r); ev = sim.get("events", {}).get(r, {})
            out[r] = {"done": len(done), "running": [str(w.c["procedure_group"].iloc[g]) for g in running], "running_g": (int(running[0]) if running else None), "waiting": [(str(w.c["case_id"].iloc[g])[:8], str(w.c["procedure_group"].iloc[g]), round(float(w.gmed[g]))) for g in waiting],
                      "finish": None if f is None else float(np.mean(f)), "p90": None if f is None else float(np.percentile(f, 90)), "events": ev}
        return out

    def _ask_llm(self, role, facts, question):
        msgs = [{"role": "system", "content": SYS.format(role=role)}, {"role": "user", "content": f"Facts you know: {facts}\nCoordinator asks: {question}"}]
        r = chat(msgs, max_tokens=220, temperature=0.0, extra=THINK_OFF, retries=5, timeout=300); self.calls += 1   # a transport failure voids the day (redone later): it must never count as a person's neutral answer
        try:
            m = re.search(r"\{.*\}", r["content"], re.S); j = json.loads(m.group(0)) if m else {"answer": r["content"][:200], "stance": "neutral", "reason": ""}
        except Exception:
            j = {"answer": r["content"][:200], "stance": "neutral", "reason": "reply not parsed"}
        if not isinstance(j, dict): j = {"answer": str(j)[:200], "stance": "neutral", "reason": "reply not parsed"}
        j["stance"] = j.get("stance", "neutral") if j.get("stance") in ("agree", "refuse", "neutral") else "neutral"
        return j

    # ---- the people's verdict (environment, every agent): asked by the harness before an action is executed ----
    def consult(self, act: dict, st: dict, t: float, sim: dict):
        """Obtain fresh binding answers for the exact candidate schedule and forecast.

        Informational dialogue is not execution consent. Each execution attempt
        is consulted again, including when another action has changed the state.
        """
        from .actions import apply, room_of
        if act.get("type") not in ("move", "swap"):
            return None, None, None
        candidate = apply(self.w, st, act, t)
        cases = [int(act["case"])] if act["type"] == "move" else [int(act["case_a"]), int(act["case_b"])]
        for g in cases:
            r = room_of(candidate, g)
            cid = str(self.w.c["case_id"].iloc[g])[:8]
            for who, code, label in (("nurse", "N1", f"team of room {r}"),
                                     ("anaesthesia", "A1", f"anaesthetist of room {r}"),
                                     ("surgeon", "S2", "surgeon")):
                answer = self.ask(who, candidate, t, sim,
                                  f"Do you accept this candidate schedule for case {cid} in room {r}?",
                                  room=r, case=g)
                if answer.get("stance") == "refuse":
                    return code, label, answer.get("answer", "")
        return None, None, None

    def people(self, act: dict, st: dict, t: float, sim: dict):
        """Binding move/swap verdict using the full candidate forecast."""
        code, who, ans = self.consult(act, st, t, sim)
        if code: return False, code, f"the {who} declines: {ans}"
        return True, "ok", ""

    def _anaes_team(self, st, r, t):
        """the anaesthesia team of room r at time t (of the running case, else the next waiting one, else the last one; hashed id, from the day's recorded roster)"""
        w = self.w; c = w.c
        if "anes_team_id" not in c: return None
        known = [g for g in st["assign"].get(r, []) if w.arrival.get(g, -1) <= t]
        running = [g for g in known if w.entry(g) <= t and float(w.clk["EXIT"][g]) > t]; waiting = [g for g in known if w.entry(g) > t]
        pick = (running or waiting or known[-1:] or [None])[0]
        team = None if pick is None else str(c["anes_team_id"].iloc[pick])
        return None if team in (None, "", "nan", "None", "<NA>") else team

    # ---- the agents ----
    def ask(self, who: str, st: dict, t: float, sim: dict, question: str, room=None, surgeon=None, case=None, dry: bool = False) -> dict:
        """dry=True returns the persona and the facts it would be given, without asking (used by the role-agent audit)"""
        w = self.w; who = str(who or "").lower(); room = _int(room)
        if who in ("nurse", "surgeon", "anaesthesia", "transport"): sim = self._with_events(st, t, sim)
        rf = self._room_facts(st, t, sim); ev = sim.get("events", {})
        if who in ("nurse", "surgeon", "anaesthesia") and case is None: case = self._case_in(st, question)
        if who == "surgeon" and room is None:                     # the room the agent's question names, if it did not pass it as an argument
            m = re.search(r"room\s*(\d+)", str(question)); room = _int(m.group(1)) if m else None
        if who == "nurse":
            if room is None or int(room) not in rf: return {"error": "nurse(room) needs a room of the suite", "stance": "neutral", "answer": ""}
            r = int(room); F = rf[r]; status = self.roles.reports(w, self.day, st, t, {rr: v["finish"] for rr, v in rf.items() if v["finish"] is not None}).get(r, {}).get("status", "")
            proposal = ""
            if case is not None and int(case) in ev.get(r, {}):
                S, E, fin = self._window(ev, rf, case, r)
                proposal = f" In this schedule, case {str(w.c['case_id'].iloc[int(case)])[:8]} ({w.c['procedure_group'].iloc[int(case)]}) occupies your room from {hhmm(S)} to {hhmm(E)}; the room finishes at about {hhmm(fin)}, {'within' if fin <= LATE + 30 else 'past'} that limit."
            facts = f"Room {r}: {F['done']} cases done, running: {F['running'] or 'none'}, waiting: {F['waiting'] or 'none'}; OR-GPT expects the room to finish at {hhmm(F['finish'])} (90th percentile {hhmm(F['p90'])}); the team's status: {status}. The team refuses a case that would push its finish beyond {hhmm(LATE + 30)}.{proposal}"
            role = f"circulating nurse of operating room {r}"
        elif who == "surgeon":
            sid = str(surgeon or (w.surgeon[int(case)] if case is not None else ""))
            if not sid: return {"error": "surgeon(surgeon) needs a surgeon id or a case", "stance": "neutral", "answer": ""}
            items = self._worklist(st, t, ev, lambda g: str(w.surgeon[g]) == sid, surgery=True); schedule = self._render(items)
            win = self.roles.window.get(sid, self.roles.default_window); proposal = ""
            if case is not None and room is not None and int(case) in ev.get(int(room), {}):
                g = int(case); S, E, _ = self._window(ev, rf, g, int(room), surgery=True); clash = [x for x in items if x["g"] != g and x["in"] < E and x["out"] > S]
                proposal = f" In this schedule, case {str(w.c['case_id'].iloc[g])[:8]} in room {int(room)} would run from about {hhmm(S)} to {hhmm(E)}, starting {'before' if S <= win else 'after'} your usual end of day" + (f"; in that time you are also in: {self._render(clash)}." if clash else "; that does not overlap any of your other cases.")
            facts = f"You are surgeon {sid[:6]}. Your remaining cases today, with the times OR-GPT expects: {schedule}. In the past your last case of a day usually ended by {hhmm(win)}; you refuse to start a case after that. You cannot be in two rooms at the same time, but you can operate in another room at a time when you have no other case.{proposal}"
            role = f"surgeon {sid[:6]}"
        elif who == "anaesthesia":
            if room is None or int(room) not in rf: return {"error": "anaesthesia(room) needs a room", "stance": "neutral", "answer": ""}
            r = int(room); nxt = [g for g in st["assign"][r] if w.entry(g) > t and w.arrival.get(g, -1) <= t]
            team = self._anaes_team(st, r, t); items = [] if team is None else self._worklist(st, t, ev, lambda g: str(w.c["anes_team_id"].iloc[g]) == team)
            roster = "The roster does not say which cases your team has today." if team is None else f"Your team ({team[:6]}) today, with the times OR-GPT expects: {self._render(items)}."
            rule = f"You refuse a case if your team would be in another room during that time, or if it would start after {hhmm(LATE + 30)}."
            arriving = ""
            if case is not None and int(case) in ev.get(r, {}):
                ga = int(case); S, E, _ = self._window(ev, rf, ga, r); clash = [x for x in items if x["g"] != ga and x["room"] != r and x["in"] < E and x["out"] > S]
                arriving = f" The case you are asked about: {str(w.c['case_id'].iloc[ga])[:8]} ({w.c['procedure_group'].iloc[ga]}), anaesthesia type {w.c['anesthesia_type'].iloc[ga]}, age {w.age[ga]:.0f}, about {E - S:.0f} min; it would be in your room from about {hhmm(S)} to {hhmm(E)}, starting {'before' if S <= LATE + 30 else 'after'} {hhmm(LATE + 30)}" + ("" if team is None else f"; in that time your team is also due in: {self._render(clash)}." if clash else "; your team has no other case in that time.") + ("." if team is None else "")
            if not nxt: facts = f"Room {r} has no further case today; OR-GPT expects it to finish at {hhmm(rf[r]['finish'])}. {roster} {rule}{arriving}"
            else:
                g = nxt[0]; e = rf[r]["events"].get(int(g)); c = w.c.iloc[g]
                facts = f"Room {r}, next case {str(c['case_id'])[:8]} ({c['procedure_group']}), anaesthesia type {c.get('anesthesia_type', 'unknown')}, age {w.age[g]:.0f}. OR-GPT expects the patient in the room at {hhmm(e[0]) if e else 'unknown'} and incision at {hhmm(e[1]) if e else 'unknown'} (induction and preparation {max(0, e[1] - e[0]):.0f} min). {roster} {rule}{arriving}" if e else f"Room {r}, next case {str(c['case_id'])[:8]} ({c['procedure_group']}), anaesthesia type {c.get('anesthesia_type', 'unknown')}; OR-GPT expects the room to finish at {hhmm(rf[r]['finish'])}. {roster} {rule}{arriving}"
            role = f"anaesthetist of operating room {r}"
        elif who == "transport":
            if room is None or int(room) not in rf: return {"error": "transport(room) needs a room", "stance": "neutral", "answer": ""}
            r = int(room); nxt = [g for g in st["assign"][r] if w.entry(g) > t and w.arrival.get(g, -1) <= t]
            if not nxt: facts = f"Room {r} has no further patient to bring today."
            else:
                g = nxt[0]; e = rf[r]["events"].get(int(g)); c = w.c.iloc[g]
                facts = f"Next patient for room {r}: case {str(c['case_id'])[:8]} from {c['department']}; the patient is expected in the room at {hhmm(e[0]) if e else 'unknown'}. It is now {hhmm(t)}."
            role = "ward and patient-transport coordinator"
        elif who == "ed":
            facts = (f"An emergency is arriving: case {self.info['case_id'][:8]}, {self.info['procedure_group']}, expected {self.info['expected_min']:.0f} min, arrived at {hhmm(self.info['arrival'])}, placed next in room {self.info['room']}." if self.info and self.info.get("arrival", 1e9) <= t else "No emergency has arrived so far today.")
            role = "emergency department physician"
        elif who == "patient":
            if case is None: return {"error": "patient(case) needs a case", "stance": "neutral", "answer": ""}
            g = int(case); c = w.c.iloc[g]
            facts = f"You are the patient of elective case {str(c['case_id'])[:8]}: age {w.age[g]:.0f}, {c['department']}, procedure {c['procedure_group']}, scheduled today. (Simulated role; no patient preferences are supplied.)"
            role = "patient waiting for surgery today"
        else: return {"error": "unknown role; use nurse, surgeon, anaesthesia, transport, ed or patient", "stance": "neutral", "answer": ""}
        if dry: return {"role": role, "facts": facts}
        j = self._ask_llm(role, facts, question)
        entry = {"t": t, "who": who, "room": room, "surgeon": (surgeon or None), "case": case, "question": question[:300], "answer": j.get("answer", "")[:300], "stance": j["stance"], "reason": j.get("reason", "")[:200], "facts": facts[:700]}
        self.log.append(entry); return {"role": role, "answer": entry["answer"], "stance": entry["stance"]}
