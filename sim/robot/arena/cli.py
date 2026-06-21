# SPDX-License-Identifier: MIT
"""`arena` unified entrypoint — the single command that replaces curriculum_drive / selfplay_drive
and the /tmp/rp_*.sh orchestration.

  # skill curriculum then self-play, on the local CPU venv (validation):
  python -m arena.cli pipeline --seed curriculum_best.pkl --runner local --lean --tiny
  # real run on a rented A100 (Phase 8):
  python -m arena.cli pipeline --seed curriculum_best.pkl --runner pod --gpu "NVIDIA A100 80GB PCIe" \
      --envs 8192 --lean --steps-per-phase 10000000 --round-steps 10000000 --budget 25

  python -m arena.cli --selftest
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path

from arena.schedule import Curriculum, League, Pipeline   # noqa: E402
from arena.runner import LocalRunner, Pod, PodRunner       # noqa: E402
from arena.run import Run                                  # noqa: E402
from arena.coach import Coach                              # noqa: E402


def build_schedule(args):
    cur = Curriculum(steps_per_phase=args.steps_per_phase)
    lg = League(seed=args.seed, rounds=args.rounds, round_steps=args.round_steps)
    base = cur if args.cmd == "curriculum" else lg if args.cmd == "league" else Pipeline([cur, lg])
    return Coach.default(base) if args.coach else base     # --coach: auto-adapt reward weights to laggards


def build_runner(args):
    if args.runner == "local":
        return LocalRunner(kernel=args.kernel, out=os.environ.get("CODESIGN_OUT", "."),
                           lean=args.lean, envs=args.envs, batch=args.batch, tiny=args.tiny)
    return PodRunner(Pod(), out="/root/proj/out", lean=args.lean, envs=args.envs,
                     batch=args.batch, tiny=args.tiny)      # pod provision/teardown handled by Run.go (Phase 8)


def make_parser():
    ap = argparse.ArgumentParser(prog="arena")
    ap.add_argument("cmd", choices=["curriculum", "league", "pipeline"], help="which schedule")
    ap.add_argument("--seed", default=None, help="seed ckpt for the league (the skill fighter)")
    ap.add_argument("--steps-per-phase", type=int, default=10_000_000)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--round-steps", type=int, default=10_000_000)
    ap.add_argument("--runner", choices=["local", "pod"], default="local")
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe")
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--envs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--coach", action="store_true", help="auto-adapt reward weights to lagging competencies")
    ap.add_argument("--name", default="arena-run")
    ap.add_argument("--rundir", default=None)
    ap.add_argument("--kernel", default=None)
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    run = Run(args.name, build_schedule(args), build_runner(args), rundir=args.rundir)
    run.go()
    png = run.figure()
    print(f"arena: {args.name} done — best_bench={run.state.best_bench:.2f} "
          f"best_ckpt={run.state.best_ckpt} figure={png} errors={len(run.errors())}", flush=True)
    return run


def _selftest():
    import tempfile
    from arena.run import _STUB
    tmp = Path(tempfile.mkdtemp())
    stub = tmp / "stub.py"; stub.write_text(_STUB)
    os.environ["CODESIGN_OUT"] = str(tmp / "out")
    seed = tmp / "out" / "seed.pkl"; seed.parent.mkdir(parents=True, exist_ok=True); seed.write_bytes(b"s")
    run = main(["pipeline", "--seed", str(seed), "--runner", "local", "--lean", "--kernel", str(stub),
                "--steps-per-phase", "1000", "--round-steps", "1000", "--rounds", "2",
                "--name", "cli", "--rundir", str(tmp / "rd")])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import curriculum_drive as cd
    assert run.state.completed == [p["name"] for p in cd.PHASES] and run.state.round == 2
    assert run.figure() and Path(run.figure()).exists()
    print("PROVEN: arena.cli unified entrypoint runs curriculum/league/pipeline end-to-end")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import tempfile
        os.environ.setdefault("CODESIGN_OUT", tempfile.mkdtemp())
        _selftest()
    else:
        main()
