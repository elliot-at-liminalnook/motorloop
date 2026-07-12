# SPDX-License-Identifier: MIT
"""EXTREMELY lightweight end-to-end harness for the GPU co-design loop.

Purpose (per Elliot): prove every link of the loop works *at all* at micro-scale, and
*profile where wall-clock goes* so we can see which stage is the bottleneck as the loop
evolves. This is a LIVING script — extend it as each checklist phase lands (add a stage,
add metrics). It is NOT a training run; budgets are tiny (`--tiny` in each sub-script).

The loop it exercises (each a fresh process = realistic isolation + compile cost):
  1. universal_train  grouped-design MuJoCo-Warp PPO
  2. universal_eval   deterministic evaluation of its Torch checkpoint
  3. combat_train     fused two-robot MuJoCo-Warp PPO
  4. combat_eval      deterministic combat evaluation

Per stage it records: wall_s, return code, every METRIC line the sub-script emits
(compile_s, train_s, throughput, rewards, rho, ...), and peak GPU mem/util (sampled
every 0.5 s). Writes:
  <OUT>/e2e_metrics.json    last run, full structure
  <OUT>/e2e_history.jsonl   one JSON record appended per run (schema-evolution proof)
and prints a stage-by-stage timing table (the bottleneck view).

  CODESIGN_OUT=/root/proj/out python3 e2e.py            # full tiny loop
  python3 e2e.py --only universal_train,combat_train     # subset
"""

from __future__ import annotations

import argparse, json, os, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CODESIGN_OUT", "/root/proj/out")); OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable


class GpuSampler(threading.Thread):
    """Poll nvidia-smi every 0.5 s for the duration of a stage; keep the peak."""
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = False; self.max_mem = 0; self.max_util = 0; self.available = False

    def run(self):
        while not self.stop:
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                mem, util = (int(x) for x in o.stdout.strip().splitlines()[0].split(","))
                self.available = True
                self.max_mem = max(self.max_mem, mem); self.max_util = max(self.max_util, util)
            except Exception:
                pass
            time.sleep(0.5)


def run_stage(name, cmd, env):
    print(f"\n{'='*70}\n[e2e] STAGE {name}: {' '.join(cmd)}\n{'='*70}", flush=True)
    samp = GpuSampler(); samp.start()
    t = time.time()
    proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, metrics = [], {}
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        line = line.rstrip("\n"); lines.append(line)
        if line.startswith("METRIC "):
            try:
                d = dict(kv.split("=", 1) for kv in line[len("METRIC "):].split())
                metrics[d.pop("stage", "?")] = d
            except Exception:
                pass
    proc.wait(); samp.stop = True; samp.join(timeout=2)
    wall = time.time() - t
    rec = {"name": name, "cmd": " ".join(cmd), "wall_s": round(wall, 1),
           "rc": proc.returncode, "ok": proc.returncode == 0, "metrics": metrics,
           "gpu_peak_mb": samp.max_mem if samp.available else None,
           "gpu_peak_util": samp.max_util if samp.available else None,
           "tail": lines[-30:] if proc.returncode != 0 else lines[-2:]}
    flag = "OK " if rec["ok"] else "FAIL"
    print(f"[e2e] STAGE {name} {flag} in {wall:.1f}s  gpu_peak={rec['gpu_peak_mb']}MB", flush=True)
    return rec


def gpu_info():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        return o.stdout.strip().splitlines()[0]
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of stages to run")
    ap.add_argument("--skip", default="", help="comma list of stages to skip")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="abort at first failing stage (default: keep going, record all)")
    args = ap.parse_args()

    env = dict(os.environ, CODESIGN_OUT=str(OUT))
    universal = str(OUT / "e2e_universal")
    combat = str(OUT / "e2e_combat")
    all_stages = [
        ("universal_train", [PY, "-u", "codesign_gpu.py", "--tiny", "--tag", universal]),
        ("universal_eval", [PY, "-u", "warp_eval.py", "eval", "--geometry", "universal",
                            "--checkpoint", universal + ".pt", "--episodes", "1",
                            "--steps", "16", "--envs", "4"]),
        ("combat_train", [PY, "-u", "train_adversarial.py", "--tiny", "--tag", combat]),
        ("combat_eval", [PY, "-u", "warp_eval.py", "eval", "--geometry", "combat",
                         "--checkpoint", combat + ".pt", "--episodes", "1",
                         "--steps", "16", "--envs", "4"]),
    ]
    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    stages = [(n, c) for n, c in all_stages if (not only or n in only) and n not in skip]

    t_all = time.time()
    records = []
    for name, cmd in stages:
        rec = run_stage(name, cmd, env)
        records.append(rec)
        if not rec["ok"] and args.stop_on_fail:
            print(f"[e2e] stopping: {name} failed (--stop-on-fail)", flush=True)
            break
    total_s = round(time.time() - t_all, 1)

    run = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gpu": gpu_info(), "python": sys.version.split()[0],
           "total_s": total_s, "n_stages": len(records),
           "all_ok": all(r["ok"] for r in records) and len(records) == len(stages),
           "stages": records}

    # load the previous run BEFORE appending this one (cross-run trend view)
    hist_path = OUT / "e2e_history.jsonl"
    prev = None
    if hist_path.exists():
        try:
            lines = [l for l in hist_path.read_text().splitlines() if l.strip()]
            if lines: prev = json.loads(lines[-1])
        except Exception:
            prev = None
    prev_wall = {s["name"]: s["wall_s"] for s in (prev.get("stages", []) if prev else [])}

    (OUT / "e2e_metrics.json").write_text(json.dumps(run, indent=2))
    with open(hist_path, "a") as fh:
        fh.write(json.dumps(run) + "\n")

    # ---- bottleneck view (Δ = wall-time change vs the previous run in e2e_history.jsonl) ----
    print(f"\n{'='*78}\n[e2e] TIMING  (total {total_s}s, gpu {run['gpu']})\n{'='*78}")
    print(f"  {'stage':<12}{'wall_s':>9}{'Δprev':>8}{'%tot':>7}{'compile':>9}{'gpu_MB':>9}  key")
    for r in records:
        m = r["metrics"]
        # pull the most informative per-stage number(s)
        key = ""
        if r["name"] == "universal_train":
            key = (f"reward={m.get('walker_train',{}).get('final_reward','?')} "
                   f"rho={m.get('phase2_corr',{}).get('rho','?')} "
                   f"cem={m.get('cem',{}).get('best','?')}")
        elif r["name"] == "universal_eval":
            key = f"bodies={m.get('build_pack',{}).get('n_bodies','?')}"
        elif r["name"] == "combat_train":
            ft = m.get("fighter_train", {})
            key = f"sparc={ft.get('final_sparc','?')} warm={ft.get('warm','?')}"
        elif r["name"] == "combat_eval":
            sf = m.get("score_fighter", {})
            key = f"rho={sf.get('rho','?')} wb_rank={sf.get('walker_best_rank','?')}"
        # compile_s lives under different sub-stage keys; grab the largest reported
        comp = max([float(v.get("compile_s", 0)) for v in m.values() if "compile_s" in v] + [0.0])
        pct = 100 * r["wall_s"] / total_s if total_s else 0
        delta = f"{r['wall_s'] - prev_wall[r['name']]:+.1f}" if r["name"] in prev_wall else "  --"
        st = "OK" if r["ok"] else f"FAIL(rc={r['rc']})"
        print(f"  {r['name']:<12}{r['wall_s']:>9.1f}{delta:>8}{pct:>6.0f}%{comp:>9.1f}"
              f"{(r['gpu_peak_mb'] or 0):>9}  {st}  {key}")
    print(f"\n[e2e] {'ALL STAGES OK' if run['all_ok'] else 'SOME STAGES FAILED'} — "
          f"history: {OUT/'e2e_history.jsonl'}")
    sys.exit(0 if run["all_ok"] else 1)


if __name__ == "__main__":
    main()
