"""Train and simulate using the companion repository's entirely fictional tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from orgpt.config import Site
from orgpt.features import FeatureSpace
from orgpt.sites import load_site
from orgpt.train import train_model
from virtual_suite import actions
from virtual_suite.world import World


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "OR-GPT/sample_data/synthetic")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; configure CUDA_VISIBLE_DEVICES and a compatible PyTorch build.")
    torch.set_num_threads(1)
    site = Site("synthetic", args.data_dir / "cases.parquet",
                args.data_dir / "consecutive_case_pairs.parquet",
                (2030,), (2030, 2031), 2032, "Entirely fictional demonstration")
    data = load_site(site)
    assert data.cases.case_id.str.startswith("SYNTHETIC_").all()
    space = FeatureSpace().fit([data])
    encoded = space.transform(data)
    model, _ = train_model(encoded.collect("tr", "rand", args.device),
                           encoded.collect("va", 0.5, args.device),
                           encoded.NUMTOK, space.nemb, 0, args.device,
                           max_epochs=1, patience=1, d=32, heads=4, layers=1)
    world = World(data, space, model, device=args.device)
    day = world.days()[0]
    state = actions.init_state(day)
    cutoff = 11 * 60
    source, target = sorted(day.rooms)
    waiting = actions.remaining(world, state, source, cutoff)
    action = {"type": "move", "case": waiting[-1], "to": target, "position": "end"}
    permitted, code, _ = actions.gate(world, day, state, action, cutoff)
    assert permitted and code == "ok"
    candidate = actions.apply(world, state, action, cutoff)
    for schedule in (state, candidate):
        result = world.simulate(day, schedule["assign"], cutoff, n=2,
                                rng=np.random.default_rng(0), detail=True)
        assert np.isfinite(result["overtime_min"]).all()
        assert all(np.isfinite(finish).all() for finish in result["finish"].values())
    # One realised trajectory is kept between decisions. Forecast samples above
    # do not become the environment's realised future.
    episode = world.start_episode(day, seed=7)
    permitted, code, _ = actions.gate(episode, episode.day, state, action, cutoff)
    assert permitted and code == "ok"
    episode.commit(state, candidate, cutoff)
    episode.advance(13 * 60)
    observed = {ev: clocks.copy() for ev, clocks in episode.clk.items()}
    episode.simulate(episode.day, candidate["assign"], 13 * 60, n=2,
                     rng=np.random.default_rng(8))
    for ev in observed:
        np.testing.assert_array_equal(observed[ev], episode.clk[ev])
    completed = episode.finish()
    for events in completed["events"].values():
        assert np.all(np.diff(np.array(list(events.values())).reshape(-1)) >= 0)
    world.age[action["case"]] = 80.0
    permitted, code, _ = actions.gate(world, day, state, action, cutoff)
    assert not permitted and code == "G5"
    # A separate fictional day starts before either room is occupied and gives
    # their first cases the same surgeon. Each realised interval must be disjoint.
    world.surgeon = world.surgeon.copy()
    first, second = day.rooms[source][0], day.rooms[target][0]
    world.surgeon[second] = world.surgeon[first]
    coupled = world.start_episode(day, cutoff=7 * 60, seed=9).finish()
    a = coupled["events"][source][first][1:3]
    b = coupled["events"][target][second][1:3]
    assert max(a[0], b[0]) >= min(a[1], b[1])
    print(f"Device: {next(model.parameters()).device}")
    print(f"Loaded {len(data.cases)} fictional cases; trained for one epoch.")
    print("Forecast, committed move, virtual progression, shared-surgeon occupancy and protected-case refusal: passed.")
    print("The action is executed in the fictional virtual day only.")


if __name__ == "__main__":
    main()
