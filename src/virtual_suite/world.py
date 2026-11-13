"""OR-GPT suite simulation from explicitly supplied data, feature space and model.

World(site_data, feature_space, model, device="cuda") creates no data files and
loads no hospital configuration or checkpoints. Shared surgeons couple room trajectories.
"""
from __future__ import annotations
import heapq
import numpy as np, pandas as pd, torch
from orgpt import config as C
from orgpt.sites import SiteData
from orgpt.features import EVT, f32, i64, with_time_of_day
from orgpt.sampling import draw_duration

LV = np.array(C.QS); LATE = 17.5 * 60


class Day:
    """One calendar day of one campus: rooms -> ordered case indices (into world.c), with the observed clock times."""
    def __init__(self, date, rooms: dict, world):
        self.date, self.rooms, self.w = date, rooms, world
        self.n_pairs_day = sum(max(len(gs) - 1, 0) for gs in rooms.values())

    def entered(self, room, cutoff):
        return [g for g in self.rooms[room] if self.w.entry(g) <= cutoff]

    def remaining(self, room, cutoff):
        return [g for g in self.rooms[room] if self.w.entry(g) > cutoff]


class World:
    def start_episode(self, day, cutoff=660.0, *, seed=0):
        """Initialise a continuous virtual day from observations through cutoff."""
        from .trajectory import EpisodeWorld
        return EpisodeWorld(self, day, cutoff, seed=seed)

    def __init__(self, site_data: SiteData, feature_space, model, device: str = "cuda"):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; configure a GPU and a compatible PyTorch build.")
        self.site = site_data.name; self.dev = device
        self.sd = site_data; self.sp = feature_space
        self.enc = self.sp.transform(self.sd)
        self.m = model.to(device); self.m.eval()
        self.c = self.enc.c; e = self.enc
        base = self.c.groupby("room_day")["or_entry"].transform("min")
        clock0 = ((base - base.dt.normalize()).dt.total_seconds() / 60.0).to_numpy(f32)
        self.clk = {ev: e.ab[ev] + clock0 for ev in EVT}  # preserve exits after midnight, rather than wrap to 00:xx
        self.mu_p, self.sd_p = self.sp.rec_stats["prior"]; self.des_dim = e.des_dim
        dev = self.c[self.c.yr.isin(self.sd.site.dev_years)]
        self.room_cap = {r: set(g["procedure_group"].astype(str)) for r, g in dev.groupby("room")}       # procedure groups a room has hosted
        self.gmed = self.c["procedure_group"].map(self.sp.gmed).fillna(self.sp.gmed_default).to_numpy(float)
        self.surgeon = self.c["surgeon_id"].astype(str).to_numpy(); self.age = pd.to_numeric(self.c["age"], errors="coerce").to_numpy(float)
        self.emergency = (self.c["surgery_type"].astype(str) == "emergency").to_numpy() if "surgery_type" in self.c else np.zeros(len(self.c), bool)
        self.day_of = pd.to_datetime(self.c["or_entry"]).dt.normalize()
        self._pair_cache = {}; self.bias = 1.0      # robustness: multiply every sampled interval (misspecified evaluation world)
        self.arrival = {}; self.forced_emergency = set()   # scenario: cases injected into a day arrive at a clock (not their recorded entry) and are emergencies

    def entry(self, g):
        """recorded entry clock of a case. A case injected into the simulation has no entry in the record: it never counts as started in
        a replayed view (its recorded clocks belong to another day), and its timing comes only from simulation, never before its arrival"""
        return float("inf") if g in self.arrival else float(self.clk["ENTRY"][g])

    def known(self, g, t):
        """whether the coordinator can know of a case at clock t: every recorded case, and an injected one once it has arrived"""
        return self.arrival.get(g, -1.0) <= t

    def is_emergency(self, g):
        return bool(self.emergency[g]) or g in self.forced_emergency

    # ---- days ----
    def days(self, year=None):
        year = self.sd.site.test_year if year is None else year
        out = []
        for d, rows in self.c[self.c.yr == year].groupby(self.day_of):
            rooms = {int(r): g.sort_values("or_entry").index.to_numpy() for r, g in rows.groupby("room")}
            out.append(Day(str(d.date()), {r: [int(x) for x in gs] for r, gs in rooms.items()}, self))
        return out

    def day(self, date):
        return next(d for d in self.days(year=pd.Timestamp(date).year) if d.date == date)

    # ---- reconfiguration block for ANY (prev, next) pair ----
    def pair_rec(self, gp, gn, prior_turn, rv, dv, pdx, *, surgery_end):
        """Static case attributes + this trajectory's realised surgery-end clock.

        The time-of-day band in training is measured at SEND (not EXIT).
        Never cache a dynamic time band or consult a future recorded timestamp.
        """
        key = (gp, gn)
        if key not in self._pair_cache:
            P, N = self.c.iloc[gp], self.c.iloc[gn]
            gt = lambda o, k, d="": o[k] if k in o.index and pd.notna(o[k]) else d
            row = dict(prev_procedure=gt(P, "procedure"), next_procedure=gt(N, "procedure"), prev_proc_group=gt(P, "procedure_group"), next_proc_group=gt(N, "procedure_group"),
                       prev_anes=gt(P, "anesthesia_type", "missing"), next_anes=gt(N, "anesthesia_type", "missing"), prev_grade=gt(P, "surgery_grade", "missing"), next_grade=gt(N, "surgery_grade", "missing"),
                       prev_type=gt(P, "surgery_type", "missing"), next_type=gt(N, "surgery_type", "missing"), same_department=bool(gt(P, "department") == gt(N, "department")),
                       same_surgeon=bool(str(gt(P, "surgeon_id")) == str(gt(N, "surgeon_id"))), next_asa=gt(N, "asa", np.nan), tod="missing")
            # Original pairs carry nursing-set overlap. For a novel pair, identical known
            # nursing-team ids establish overlap; anaesthesia teams are a different feature.
            nurses = (str(gt(P, "nursing_team_id")), str(gt(N, "nursing_team_id")))
            row["team_overlap"] = nurses[0] == nurses[1] and nurses[0] not in ("", "None", "nan", "missing")
            pairs = getattr(self.sd, "pairs", self.sd.vp)
            match = pairs[(pairs.prev_case_id.astype(str) == str(P.case_id)) & (pairs.next_case_id.astype(str) == str(N.case_id))]
            if len(match): row["team_overlap"] = bool(match.iloc[0].team_overlap)
            self._pair_cache[key] = self.sp.bridge.design_matrix(self.sp.bridge.transform(pd.DataFrame([row])), intercept=False)[0][0].astype(f32)
        des = with_time_of_day(self._pair_cache[key], self.des_dim, surgery_end)
        rs = self.sp.rec_stats
        z = lambda k, a: (np.log1p(max(a, 0.0)) - rs[k][0]) / rs[k][1] if k != "pd" else (a - rs[k][0]) / rs[k][1]
        prior = 0.0 if prior_turn is None else (np.log1p(max(prior_turn, 0.0)) - self.mu_p) / self.sd_p
        return np.concatenate([des, np.array([prior, z("rv", rv), z("dv", dv), z("pd", pdx)], f32)]).astype(f32)

    # ---- simulation ----
    def simulate(self, day: Day, assignment: dict, cutoff: float, n: int = 20, rng=None, late=LATE, detail=False):
        """Play the complete suite forward from `cutoff` (clock minutes).
        assignment must include all rooms; started cases keep their current room/order.
        Returns per room an (n,) array of finish clocks and the suite aggregates. Injected cases that
        arrive after the cutoff are unknown to whoever forecasts at that time and are left out."""
        current = self._assignment if getattr(self, "is_episode", False) else day.rooms
        if set(assignment) != set(current):
            raise ValueError("Forecast the complete suite to retain shared-surgeon constraints")
        for room, gs in current.items():
            started = [g for g in gs if self.entry(g) <= cutoff]
            if [g for g in assignment[room] if self.entry(g) <= cutoff] != started:
                raise ValueError("Forecast must preserve started cases in their current room/order")
        rng = rng or np.random.default_rng(0); fin = {}; ev = {}; samples = {}
        visible = {room: [g for g in cases if self.known(g, cutoff)] for room, cases in assignment.items()}
        order, trajectories = self._suite_trajectories(day, visible, cutoff, n, rng)
        for room, T in trajectories.items():
            fin[room] = T[:, -1, 3].copy()
            if detail:
                samples[room] = {int(g): T[:, k, :].copy()
                                 for k, g in enumerate(order[room])}
                ev[room] = {int(g): T[:, k, :].mean(0).tolist()
                            for k, g in enumerate(order[room])}
        F = np.stack([fin[r] for r in fin]) if fin else np.zeros((0, n))
        over = np.clip(F - late, 0, None).sum(0); nlate = (F > late).sum(0)
        out = {"finish": fin, "overtime_min": over, "late_rooms": nlate, "makespan": F.max(0) if len(F) else np.zeros(n)}
        if detail: out["events"] = ev          # room -> case -> expected [entry, surgery start, surgery end, exit] clocks (mean over continuations)
        if detail: out["event_samples"] = samples  # same keys, (n, 4) clocks for safety checks
        return out

    def _surgeon_key(self, g):
        surgeon = str(self.surgeon[g]).strip()
        if surgeon.lower() in {"", "nan", "none", "<na>", "missing", "unknown"}:
            raise ValueError(f"Case {g} requires a surgeon identifier for resource scheduling")
        return surgeon

    def _validate_surgeon_path(self, events, cutoff):
        """Surgery occupies its named surgeon on [SSTART, SEND), including overruns."""
        by_surgeon = {}
        for room, cases in events.items():
            for g, times in cases.items():
                start, end = float(times[1]), float(times[2])
                if end <= cutoff:
                    continue
                surgeon = self._surgeon_key(g)
                for other_room, other_start, other_end in by_surgeon.get(surgeon, []):
                    if room != other_room and max(start, other_start, cutoff) < min(end, other_end):
                        raise ValueError(f"Overlapping surgery for surgeon {surgeon}")
                by_surgeon.setdefault(surgeon, []).append((room, start, end))

    def _suite_trajectories(self, day, assignment, cutoff, n, rng):
        """Sample coupled room trajectories with a surgeon calendar per continuation.

        Requests are processed by their ready time, with room index as a stable tie
        break. An ongoing surgery has priority. Waiting is a lower bound on start,
        not an extra duration added to a predicted interval that may contain waiting.
        """
        if n < 1:
            raise ValueError("At least one continuation is required")
        order = {r: ([g for g in gs if self.entry(g) <= cutoff]
                     + [g for g in gs if self.entry(g) > cutoff])
                 for r, gs in sorted(assignment.items()) if gs}
        if len({g for gs in order.values() for g in gs}) != sum(map(len, order.values())):
            raise ValueError("A case cannot appear in more than one room")
        day_pairs = sum(max(len(gs) - 1, 0) for gs in order.values())
        resources = {}
        active = {}
        for room, gs in order.items():
            for g in gs:
                if self.clk["SEND"][g] <= cutoff:
                    continue
                surgeon = self._surgeon_key(g)
                resources.setdefault(surgeon, set()).add(room)
                if self.clk["SSTART"][g] <= cutoff:
                    if surgeon in active:
                        raise ValueError("Observed ongoing surgeries overlap for the same surgeon")
                    active[surgeon] = (room, g)
        # Disjoint rooms retain batched model inference; shared surgeons require
        # event-by-event interleaving so delays enter subsequent model contexts.
        if not any(len(rooms) > 1 for rooms in resources.values()):
            times = {r: self._room(day, r, gs, cutoff, n, rng,
                                   return_T=True, day_pairs=day_pairs)[1]
                     for r, gs in order.items()}
            return order, times
        times = {r: np.empty((n, len(gs), 4), dtype=float) for r, gs in order.items()}
        for sample in range(n):
            generators = {r: self._room_events(day, r, gs, cutoff, 1, rng, day_pairs)
                          for r, gs in order.items()}
            queue = []
            available = {}

            def enqueue(room, request):
                g, ev, proposed = request
                value = float(proposed[0])
                if not np.isfinite(value) or value < cutoff:
                    raise ValueError("Invalid future event time")
                heapq.heappush(queue, (value, room, g, ev))
                return g, ev, value

            # Initialise every room before processing any future request, so a
            # surgeon who is already operating cannot be allocated to another room.
            for room, generator in generators.items():
                try:
                    g, ev, value = enqueue(room, next(generator))
                    if ev == 2:
                        available[self._surgeon_key(g)] = value
                except StopIteration as complete:
                    times[room][sample] = complete.value[0]
            while queue:
                at, room, g, ev = heapq.heappop(queue)
                if ev == 1:
                    surgeon = self._surgeon_key(g)
                    ready = max(at, available.get(surgeon, at))
                    if ready > at:
                        heapq.heappush(queue, (ready, room, g, ev))
                        continue
                try:
                    ng, ne, nt = enqueue(room, generators[room].send(np.array([at])))
                    if nt < at:
                        raise ValueError("Generated events cannot move backwards")
                    if ev == 1:
                        if ng != g or ne != 2:
                            raise ValueError("Surgery start must be followed by surgery end")
                        available[surgeon] = nt
                except StopIteration as complete:
                    times[room][sample] = complete.value[0]
            self._validate_surgeon_path(
                {r: {g: times[r][sample, k] for k, g in enumerate(gs)}
                 for r, gs in order.items()}, cutoff)
        return order, times

    def _room(self, day, room, cases, cutoff, n, rng, return_T=False, day_pairs=None):
        events = self._room_events(day, room, cases, cutoff, n, rng, day_pairs)
        try:
            _, _, proposed = next(events)
            while True:
                _, _, proposed = events.send(proposed)
        except StopIteration as complete:
            times = complete.value
        finish = times[:, -1, 3].copy()
        return (finish, times) if return_T else finish

    def _room_events(self, day, room, cases, cutoff, n, rng, day_pairs=None):
        """Yield each unobserved event before encoding it into the next prediction.

        The suite scheduler can delay surgery start for a busy surgeon. The accepted
        start is encoded before predicting surgery end and all later events.
        """
        e = self.enc; m = self.m; NUM = e.NUMTOK; NC = e.CODES.shape[1]
        started = [g for g in cases if self.entry(g) <= cutoff]; cases = started + [g for g in cases if self.entry(g) > cutoff]
        k_n = len(cases); order = [(k, ev) for k in range(k_n) for ev in range(4)]
        T = np.full((n, k_n, 4), np.nan, np.float64)                      # sampled/observed clock per sample
        for k, g in enumerate(cases):
            for ev in range(4):
                v = float(self.clk[EVT[ev]][g])
                if v <= cutoff and g in started: T[:, k, ev] = v     # observed
        if not started: T[:, 0, 0] = max(self.arrival.get(cases[0], cutoff), cutoff)   # empty-room assumption: start at cutoff/arrival, never at a future actual entry
        F = np.zeros((n, len(order), NUM), f32); CD = np.zeros((n, len(order), NC), i64); TK = np.zeros((len(order),), i64)
        nxt = {cases[i]: cases[i + 1] for i in range(k_n - 1)}
        orig = {g: i for i, g in enumerate(day.rooms.get(room, []))}
        for idx, (k, ev) in enumerate(order):
            g = cases[k]; TK[idx] = ev
            if idx > 0 and np.isnan(T[0, k, ev]):                      # draw this event's time for every sample
                pk, pe = order[idx - 1]; xb = torch.tensor(F[:, :idx], device=self.dev); cb = torch.tensor(CD[:, :idx], device=self.dev)
                b = {"f": xb, "c": cb, "tk": torch.tensor(np.tile(TK[:idx], (n, 1)), device=self.dev), "m": torch.ones(n, idx, device=self.dev), "nc": torch.zeros(n, idx, NC, dtype=torch.long, device=self.dev)}
                prev_t = T[:, pk, pe]
                elapsed = np.maximum(cutoff - prev_t, 0.0)
                is_open = np.all(prev_t <= cutoff)
                query = is_open and np.any(elapsed > 0) and pe in (1, 3)
                if query:
                    # Only surgery and turnover have trained remaining-time QUERY targets.
                    qf = F[:, idx - 1:idx].copy()
                    qf[:, 0, :12] = e.fourier(np.float32(cutoff % 1440.0))
                    if pe == 1: qf[:, 0, 12 + e.CONT.shape[1]:] = 0.0
                    b["f"] = torch.tensor(np.concatenate([F[:, :idx], qf], axis=1), device=self.dev)
                    b["c"] = torch.tensor(np.concatenate([CD[:, :idx], CD[:, idx - 1:idx]], axis=1), device=self.dev)
                    b["tk"] = torch.tensor(np.tile(np.r_[TK[:idx], 4], (n, 1)), device=self.dev)
                    b["m"] = torch.ones(n, idx + 1, device=self.dev)
                    b["nc"] = torch.zeros(n, idx + 1, NC, dtype=torch.long, device=self.dev)
                with torch.no_grad(): q = m(b)[:, -1].cpu().numpy()
                if query:
                    T[:, k, ev] = cutoff + np.array([draw_duration(q[i], rng) for i in range(n)], f32) * self.bias
                else:
                    # PRE/POST have no residual head: condition their full-duration law.
                    dur = np.array([draw_duration(q[i] * self.bias, rng, elapsed=float(elapsed[i]) if is_open else None) for i in range(n)], f32)
                    T[:, k, ev] = prev_t + dur
                if is_open:
                    T[:, k, ev] = np.maximum(T[:, k, ev], np.nextafter(np.float32(cutoff), np.float32(np.inf)))
                if ev == 0 and g in self.arrival: T[:, k, ev] = np.maximum(T[:, k, ev], self.arrival[g])   # an injected case cannot enter before it arrives
            observed = g in started and self.clk[EVT[ev]][g] <= cutoff
            if not observed:
                T[:, k, ev] = yield g, ev, T[:, k, ev].copy()
            # encode this token for every sample (its time is now known)
            for i in range(n):
                clock = T[i, k, ev] % 1440.0; rec = np.zeros(e.NREC, f32)
                if ev == 3 and g in nxt:
                    gn = nxt[g]; observed_pair = not getattr(self, "is_episode", False) and (g in started) and (gn in started) and self.clk["EXIT"][g] <= cutoff and orig.get(gn, -1) == orig.get(g, -2) + 1
                    r0 = e.key2rec.get(e.room_s[g] + "|" + e.exit_s[g] + "|" + e.entry_s[gn]) if observed_pair else None
                    if r0 is not None: rec = r0.copy()
                    else:
                        prior = None if k == 0 else float(T[i, k, 0] - T[i, k - 1, 3])
                        rec = self.pair_rec(g, gn, prior, max(k_n - 1, 0), day.n_pairs_day if day_pairs is None else day_pairs, k,
                                            surgery_end=T[i, k, 2])
                F[i, idx] = np.concatenate([e.fourier(np.float32(clock)), e.CONT[g], rec])
                codes = e.CODES[g].copy()
                codes[C.CAT_COLS.index("room")] = self.sp.vocab["room"].get(str(room), 0)
                CD[i, idx] = codes
        return T

