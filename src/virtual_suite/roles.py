"""roles.py: the other agents of the virtual operating suite: one agent per operating room and one per surgeon. They are model-based agents (no
language model): each observes only what its real counterpart would know, reasons with OR-GPT's forecast of its own room, and answers the
coordinator's proposals. They make the suite an environment in which the
coordinator must negotiate, not merely optimise.

Room agent  r  — reports its status at every wake-up (expected finish and overtime from OR-GPT, cases remaining, whether it needs help or can
                 take a case) and REFUSES to receive a case that would push its own expected finish beyond LATE + ROOM_TOLERANCE minutes
                 (verdict R1): overtime may be reduced, not merely shifted onto another team.
Surgeon agent  — knows the surgeon's own availability window, estimated from the development years as the 90th percentile of the clock at which
                 that surgeon's last case of a day ended (surgeons with < MIN_DAYS days use the campus-wide value), and REFUSES a move or swap
                 whose expected start of their case falls after that window (verdict S1).
All verdicts are logged; they are advisory to nobody — a refused action is not executed.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .world import World, Day, LATE

ROOM_TOLERANCE, MIN_DAYS, HELP_MIN, SPARE_MIN = 30.0, 5, 30.0, 60.0


class Roles:
    def __init__(self, w: World):
        if getattr(w, "is_episode", False):
            raise ValueError("Fit Roles on the original World's development data before starting an episode")
        c = w.c; dev = c[c.yr.isin(w.sd.site.dev_years)].copy()
        dev["exit_clk"] = np.asarray(w.clk["EXIT"])[dev.index.to_numpy()] if len(dev) else []
        dev["day"] = pd.to_datetime(dev["or_entry"]).dt.normalize(); dev["sid"] = dev["surgeon_id"].astype(str)
        last = dev.groupby(["sid", "day"])["exit_clk"].max().reset_index()
        q = last.groupby("sid")["exit_clk"].agg(["count", lambda x: float(np.percentile(x, 90))]); q.columns = ["n", "p90"]
        self.default_window = float(np.percentile(last["exit_clk"], 90)) if len(last) else LATE + 120
        self.window = {s: (float(r.p90) if r.n >= MIN_DAYS else self.default_window) for s, r in q.iterrows()}
        self.w = w

    # ---- room agents ----
    def reports(self, w: World, day: Day, st: dict, t: float, fin: dict) -> dict:
        out = {}
        for r in sorted(st["assign"]):
            f = fin.get(r); rem = [g for g in st["assign"][r] if w.entry(g) > t and w.known(g, t)]
            if f is None: out[r] = {"status": "finished", "expected_finish": None, "overtime_min": 0.0, "remaining": 0}; continue
            over = max(0.0, f - LATE)
            status = "needs help" if over >= HELP_MIN else ("can take a case" if f <= LATE - SPARE_MIN else "on time")
            out[r] = {"status": status, "expected_finish": f"{int(f // 60):02d}:{int(f % 60):02d}", "overtime_min": round(over), "remaining": len(rem)}
        return out

    def surgeon_window(self, g) -> float:
        return self.window.get(str(self.w.surgeon[g]), self.default_window)

    def verdict(self, w: World, day: Day, st: dict, act: dict, t: float, fc_after: dict | None) -> tuple[bool, str, str]:
        """Use the complete candidate forecast, including case-level event times."""
        typ = act.get("type")
        moved = [(int(act["case"]), int(act["to"]))] if typ == "move" else ([(int(act["case_a"]), self._room(st, int(act["case_b"]))), (int(act["case_b"]), self._room(st, int(act["case_a"])))] if typ == "swap" else [])
        if not moved: return True, "ok", ""
        if fc_after is None: return False, "S1", "candidate event forecast is required"
        for g, r in moved:
            try:
                f = float(np.mean(fc_after["finish"][r]))
                events = np.asarray(fc_after["events"][r][g], dtype=float)
                if not np.isfinite(f) or events.shape != (4,) or not np.isfinite(events).all() or np.any(np.diff(events) < 0):
                    raise ValueError("invalid event times")
                start = float(events[1])
            except (KeyError, TypeError, ValueError):
                return False, "S1", f"case {g} has no valid candidate event forecast"
            if f > LATE + ROOM_TOLERANCE: return False, "R1", f"room {r} refuses: its expected finish would be {int(f // 60):02d}:{int(f % 60):02d}"
            win = self.surgeon_window(g)
            if start > win: return False, "S1", f"the surgeon of case {w.c['case_id'].iloc[g]} is not available after {int(win // 60):02d}:{int(win % 60):02d} (expected start {int(start // 60):02d}:{int(start % 60):02d})"
        return True, "ok", ""

    def cast(self, w: World, day: Day, log: list | None = None, info: dict | None = None):
        """the language personas of the day (cast.py), part of the environment: every agent's move or swap is put to the receiving room's
        nurse and anaesthetist and the case's surgeon before it is executed, whichever agent proposed it. Cached per day."""
        from .cast import Cast
        if getattr(self, "_cast_day", None) != day.date or getattr(getattr(self, "_cast", None), "w", None) is not w:
            self._cast_day, self._cast = day.date, Cast(w, day, self, [] if log is None else log, info)
        return self._cast

    @staticmethod
    def _room(st, g):
        return next((r for r, v in st["assign"].items() if g in v), None)
