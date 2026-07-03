# SPDX-License-Identifier: MIT
"""M3 benchmark — baseline (wrapper-way obs/reward round-trip) vs fused
(step + obs/reward/lidar kernels in one launch sequence / CUDA graph) on the
two-robot fight scene. The >=2x kill criterion of secret-sauce §10c compares
exactly these two modes end-to-end (physics identical, outputs identical to
float32 — test_m3_parity.py / test_m3_obsreward.py).

Prints one parseable RESULT line per mode (same convention as
bench_warp_vs_mjx.py) and, with --mode both, a ratio line carrying the kill
verdict. Steps are CONTROL steps (each = constants.FRAME_SKIP physics steps);
env_steps_per_s counts PHYSICS steps x nworld for comparability with
bench_warp_vs_mjx RESULT lines.

  # local CPU proxy (kernel path vs host round-trip; no graph capture on CPU):
  .venv-warp/bin/python sim/robot/warplayer/bench_m3.py --nworld 8 --steps 100 --mode both

  # definitive GPU ratio on a pod (A100), lidar on and off:
  .venv-warp/bin/python sim/robot/warplayer/bench_m3.py --nworld 4096 --steps 200 --mode both --device cuda:0
  .venv-warp/bin/python sim/robot/warplayer/bench_m3.py --nworld 4096 --steps 200 --mode both --device cuda:0 --no-lidar

On CUDA the fused mode captures the whole control step (physics + our
kernels) into ONE graph; the baseline captures its PHYSICS BLOCK too (the
wrapper way also jits/captures the step) but its obs/reward necessarily live
outside the graph as a host round-trip — that seam is the measured effect,
with physics launch overhead equalized on both sides.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402


def bench(mode: str, nworld: int, steps: int, warmup: int, lidar: bool,
          device: str | None, seed: int = 0,
          nconmax: int | None = None, njmax: int | None = None) -> dict:
    from warplayer.fused import FightLayer, build_fight_model

    dev = wp.get_device(device) if device else wp.get_device()
    with wp.ScopedDevice(dev):
        mjm, spec = build_fight_model(lidar=lidar, disable_sensors=(mode == "fused"))
        lay = FightLayer(nworld=nworld, mode=mode, lidar=lidar, seed=seed,
                         mjm=mjm, spec=spec, nconmax=nconmax, njmax=njmax)
        rng = np.random.default_rng(seed)
        lay.set_actions(rng.uniform(-0.3, 0.3, (nworld, lay.idx.nuA)))

        t0 = time.time()
        if dev.is_cuda:
            lay.capture()   # fused: steps+kernels in ONE graph; baseline: physics block only
        for _ in range(warmup):
            lay.step()
        wp.synchronize()
        warm_s = time.time() - t0

        t0 = time.time()
        for _ in range(steps):
            lay.step()
        wp.synchronize()
        wall = time.time() - t0
        assert np.isfinite(lay.reward.numpy()).all(), "non-finite reward after rollout"
        assert not np.isnan(lay.d.qpos.numpy()).any(), "NaN qpos after rollout"
    return dict(wall=wall, warm_s=warm_s, dev=str(dev),
                graph=dev.is_cuda, obs_dim=lay.obs_dim,
                frame_skip=lay.frame_skip)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nworld", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100, help="control steps (x FRAME_SKIP physics steps)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--mode", choices=("baseline", "fused", "both"), default="both")
    ap.add_argument("--lidar", dest="lidar", action="store_true", default=True)
    ap.add_argument("--no-lidar", dest="lidar", action="store_false")
    ap.add_argument("--device", default=None, help="warp device, e.g. 'cpu' or 'cuda:0'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nconmax", type=int, default=None, help="per-world contact-pool override")
    ap.add_argument("--njmax", type=int, default=None, help="per-world constraint-row override")
    args = ap.parse_args()

    wp.init()
    modes = ("baseline", "fused") if args.mode == "both" else (args.mode,)
    res = {}
    for mode in modes:
        r = bench(mode, args.nworld, args.steps, args.warmup, args.lidar, args.device,
                  args.seed, args.nconmax, args.njmax)
        res[mode] = r
        phys = args.nworld * args.steps * r["frame_skip"]
        print(f"RESULT bench=m3_fused scene=fight mode={mode} nworld={args.nworld} "
              f"ctrl_steps={args.steps} frame_skip={r['frame_skip']} lidar={int(args.lidar)} "
              f"obs_dim={r['obs_dim']} device={r['dev']!r} graph={int(r['graph'])} "
              f"env_steps_per_s={phys / r['wall']:.1f} "
              f"ctrl_steps_per_s={args.nworld * args.steps / r['wall']:.1f} "
              f"wall_s={r['wall']:.3f} warmup_s={r['warm_s']:.2f}", flush=True)
    if len(res) == 2:
        ratio = res["baseline"]["wall"] / res["fused"]["wall"]
        cpu = not wp.get_device(args.device).is_cuda if args.device else not wp.get_device().is_cuda
        verdict = "PASS" if ratio >= 2.0 else "FAIL"
        tag = "cpu_proxy" if cpu else "gpu"
        print(f"RESULT bench=m3_ratio scene=fight nworld={args.nworld} lidar={int(args.lidar)} "
              f"kind={tag} fused_over_baseline={ratio:.2f} kill_criterion_2x={verdict}", flush=True)
        if cpu:
            print("# NOTE: CPU proxy only — no CUDA graph on CPU; the >=2x kill criterion "
                  "is judged on the GPU ratio (see module docstring for the pod command).",
                  flush=True)


if __name__ == "__main__":
    main()
