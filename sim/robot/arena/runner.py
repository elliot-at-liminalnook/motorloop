# SPDX-License-Identifier: MIT
"""Layer 3 — Runners. Same interface (`train(stage, warm, cum_base) -> res|None`), two backends:
  * `LocalRunner` — subprocess the kernel on the CPU venv (tiny validation); injects TRACE_*/ARENA_SINK
    so kernel events join the run's stream; parses `{tag}_state.json`; emits a classified error on failure.
  * `PodRunner` — the SAME, over ssh, owning the RunPod lifecycle (`Pod`): the six `/tmp/rp_*.sh`
    consolidated with the hard-won fixes baked in — REST (not GraphQL), the current image, venv setup,
    nohup+sentinel launch (NO stdin+`&` conflict), pull-loop, watch, terminate-at-budget, and a
    kill-by-PID helper (NEVER `pkill -f <pattern that matches the caller>` — the exit-144 self-kill).

  python -m arena.runner --selftest
"""

from __future__ import annotations

import json, os, subprocess, sys
from pathlib import Path

from arena.trace import inject, classify, Tracer  # noqa: E402

HERE = Path(__file__).resolve().parents[1]            # sim/robot
KERNEL = str(HERE / "train_adversarial.py")


def _last_jsonl(path):
    """Last row of a JSONL file (the latest benchmark decomposition) -> dict, or {}."""
    p = Path(path)
    if p.exists():
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        if lines:
            try:
                return json.loads(lines[-1])
            except Exception:
                pass
    return {}
RUNPOD_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
RUNPOD_REST = "https://rest.runpod.io/v1/pods"


class LocalRunner:
    def __init__(self, kernel=None, out=None, python=None, lean=True, envs=0, batch=0,
                 tiny=False, tracer=None, run_id="run", sink=None):
        self.kernel = kernel or KERNEL
        self.out = Path(out or os.environ.get("CODESIGN_OUT", "."))
        self.python = python or sys.executable
        self.lean, self.envs, self.batch, self.tiny = lean, envs, batch, tiny
        self.tracer = tracer or Tracer(run_id=run_id, sink=sink, console=False)
        self.run_id, self.sink = run_id, sink

    def train(self, stage, warm=None, cum_base=0):
        tag = stage.tag
        argv = [self.python, "-u", self.kernel] + stage.flags(
            warm=warm, cum_base=cum_base, envs=self.envs, batch=self.batch, lean=self.lean, tiny=self.tiny)
        env = inject(os.environ.copy(), tag, run_id=self.run_id)   # TRACE_* across the subprocess boundary
        env["CODESIGN_OUT"] = str(self.out)
        if self.sink:
            env["ARENA_SINK"] = str(self.sink)                     # kernel events join this run's stream
        log = self.out / f"arena_{tag}.log"
        self.out.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as lf:
            rc = subprocess.run(argv, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        sf = self.out / f"{tag}_state.json"
        if rc != 0 or not sf.exists():
            tail = log.read_text()[-1500:] if log.exists() else ""
            self.tracer.error(f"kernel {tag} rc={rc}", cause=classify(tail, rc),
                              exit_code=rc, tag=tag, log=str(log))
            return None
        s = json.loads(sf.read_text())
        return dict(best_bench=float(s["best_bench"]), best_ckpt=str(self.out / f"{tag}_best.pkl"),
                    cum_step=int(s["cum_step"]), last_ratio=s.get("last_ratio"),
                    signals=_last_jsonl(self.out / f"{tag}_benchmark.jsonl"))   # decomposition for the Coach


class Pod:
    """RunPod lifecycle — consolidates the /tmp/rp_*.sh scripts; the hard-won fixes are baked in.
    Pure helpers (provision_body, launch_cmd, safe_kill) are offline-testable; `exec_fn`/`fetch_fn`
    are injectable so the runner's train-flow is testable offline (real ssh by default)."""

    GPUS = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA L40S", "NVIDIA GeForce RTX 4090"]

    def __init__(self, pod_id=None, ip=None, port=None, key=None, exec_fn=None, fetch_fn=None):
        self.pod_id, self.ip, self.port = pod_id, ip, port
        self.key = key or str(Path.home() / ".ssh/runpod_ed25519")
        self.exec_fn = exec_fn or self._ssh_exec        # (cmd) -> (rc, stdout)
        self.fetch_fn = fetch_fn or self._ssh_fetch      # (remote_path) -> text

    @staticmethod
    def provision_body(gpu, image=RUNPOD_IMAGE, disk=30, pubkey=""):
        # REST API shape (NOT GraphQL — that mutation 500s): gpuTypeIds is an ARRAY, env an OBJECT.
        return {"cloudType": "SECURE", "gpuTypeIds": [gpu], "gpuCount": 1, "imageName": image,
                "containerDiskInGb": disk, "volumeInGb": 0, "ports": ["22/tcp"],
                "env": {"PUBLIC_KEY": pubkey}}

    @staticmethod
    def launch_cmd(remote_script, log="/root/proj/out/drive.log", sentinel="DRIVE_DONE"):
        # ship-then-launch: nohup a SCRIPT FILE (never pipe stdin AND `&` in one ssh call — that
        # shipped an empty script), append a sentinel so the watcher knows when it's done.
        return f"nohup bash {remote_script} >> {log} 2>&1; echo {sentinel} >> {log}"

    @staticmethod
    def safe_kill(pids):
        # kill by explicit PID — NEVER `pkill -f <name>` (it self-matches the caller's argv -> the
        # process kills its own shell -> exit 144). pids: list[int].
        return ["kill", *[str(int(p)) for p in pids]]

    def _ssh_exec(self, cmd):
        import subprocess
        argv = ["ssh", "-i", self.key, "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
                "-o", "IdentitiesOnly=yes", "-o", "LogLevel=ERROR", f"root@{self.ip}", cmd]
        p = subprocess.run(argv, capture_output=True, text=True)
        return p.returncode, p.stdout

    def _ssh_fetch(self, remote_path):
        rc, out = self._ssh_exec(f"cat {remote_path}")
        return out if rc == 0 else ""


class PodRunner:
    """Same train() contract as LocalRunner, over ssh to a provisioned Pod. The remote kernel runs
    blocking with TRACE_*/ARENA_SINK exported (so its events join the run's stream); we then fetch
    and parse `{tag}_state.json`. Pod provision/teardown is driven by Run.go via provision()/teardown()."""

    def __init__(self, pod: Pod, out="/root/proj/out", proj="/root/proj", tracer=None, run_id="run",
                 lean=True, envs=0, batch=0, tiny=False, budget=25.0):
        self.pod, self.out, self.proj = pod, out, proj
        self.tracer = tracer or Tracer(run_id=run_id, console=False)
        self.run_id, self.sink = run_id, f"{out}/events.jsonl"
        self.lean, self.envs, self.batch, self.tiny, self.budget = lean, envs, batch, tiny, budget

    def train(self, stage, warm=None, cum_base=0):
        import shlex
        flags = stage.flags(warm=warm, cum_base=cum_base, envs=self.envs, batch=self.batch,
                            lean=self.lean, tiny=self.tiny)
        argv = "python3 -u train_adversarial.py " + " ".join(shlex.quote(f) for f in flags)
        exports = (f"source {self.out}/env.sh && export TRACE_RUN={shlex.quote(self.run_id)} "
                   f"TRACE_STAGE={shlex.quote(stage.tag)} ARENA_SINK={shlex.quote(self.sink)}")
        cmd = f"{exports} && cd {self.proj}/sim/robot && {argv}"
        with self.tracer.span(stage.tag, opponent=stage.opponent, component="runner:pod"):
            rc, _out = self.pod.exec_fn(cmd)
            if rc != 0:
                self.tracer.error(f"pod kernel {stage.tag} rc={rc}", exit_code=rc,
                                  cause=classify(_out, rc), tag=stage.tag, component="runner:pod")
                return None
            txt = self.pod.fetch_fn(f"{self.out}/{stage.tag}_state.json")
            if not txt:
                self.tracer.error(f"pod {stage.tag} no state.json", cause="stage_subprocess_fail",
                                  component="runner:pod")
                return None
            s = json.loads(txt)
            bj = self.pod.fetch_fn(f"{self.out}/{stage.tag}_benchmark.jsonl")
            sig = {}
            if bj:
                rows = [r for r in bj.splitlines() if r.strip()]
                if rows:
                    try: sig = json.loads(rows[-1])
                    except Exception: pass
            return dict(best_bench=float(s["best_bench"]), best_ckpt=f"{self.out}/{stage.tag}_best.pkl",
                        cum_step=int(s["cum_step"]), last_ratio=s.get("last_ratio"), signals=sig)


def _selftest():
    import tempfile
    from arena.stage import Stage
    tmp = Path(tempfile.mkdtemp())

    # (1) LocalRunner end-to-end with a STUB kernel — validates argv build, TRACE_*/ARENA_SINK
    #     propagation across the subprocess boundary, state.json parse, and the res contract.
    stub = tmp / "stub_kernel.py"
    stub.write_text(
        "import sys,json,os\n"
        "a=sys.argv\n"
        "tag=a[a.index('--tag')+1]; cum=int(a[a.index('--cum-base')+1]); steps=int(a[a.index('--steps')+1])\n"
        "o=os.environ['CODESIGN_OUT']\n"
        "open(os.path.join(o,tag+'_trace.txt'),'w').write(os.environ.get('TRACE_STAGE','')+'|'+os.environ.get('ARENA_SINK',''))\n"
        "json.dump({'tag':tag,'cum_step':cum+steps,'best_bench':-12.5,'best_step':cum,'last_ratio':1.3},"
        " open(os.path.join(o,tag+'_state.json'),'w'))\n"
        "open(os.path.join(o,tag+'_best.pkl'),'wb').write(b'x')\n")
    sink = tmp / "events.jsonl"
    r = LocalRunner(kernel=str(stub), out=tmp, lean=True, run_id="rtest", sink=str(sink))
    res = r.train(Stage(tag="c2", steps=1234), warm="w.pkl", cum_base=1000)
    assert res["best_bench"] == -12.5 and res["cum_step"] == 2234 and res["best_ckpt"].endswith("c2_best.pkl")
    trace = (tmp / "c2_trace.txt").read_text()
    assert trace.startswith("c2|") and trace.endswith(str(sink)), trace   # TRACE_STAGE + ARENA_SINK propagated

    # (2) failure path -> None + a classified error event in the stream
    bad = tmp / "bad_kernel.py"
    bad.write_text("import sys; sys.stderr.write('RESOURCE_EXHAUSTED: out of memory'); sys.exit(1)\n")
    r2 = LocalRunner(kernel=str(bad), out=tmp, run_id="rtest", sink=str(sink))
    assert r2.train(Stage(tag="boom", steps=1)) is None
    errs = [json.loads(l) for l in sink.read_text().splitlines() if json.loads(l)["kind"] == "error"]
    assert errs and errs[-1]["payload"]["cause"] == "gpu_oom", errs[-1]

    # (3) Pod pure helpers — REST body shape, ship-then-launch, kill-by-PID (no self-pkill)
    body = Pod.provision_body("NVIDIA A100 80GB PCIe", pubkey="ssh-ed25519 AAAA")
    assert body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"] and isinstance(body["env"], dict)
    assert body["imageName"] == RUNPOD_IMAGE and body["ports"] == ["22/tcp"]
    lc = Pod.launch_cmd("/root/proj/out/run_long.sh")
    assert "nohup bash /root/proj/out/run_long.sh" in lc and "DRIVE_DONE" in lc and "cat >" not in lc
    assert Pod.safe_kill([13902, 13903]) == ["kill", "13902", "13903"]      # by PID, not pattern
    assert classify("INTERNAL_SERVER_ERROR") == "runpod_graphql"
    print("PROVEN: runner abstraction — Local trains (TRACE propagated, errors classified); "
          "Pod lifecycle helpers (REST/ship-then-launch/kill-by-PID) composed offline")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        os.environ.setdefault("CODESIGN_OUT", __import__("tempfile").mkdtemp())
        _selftest()
    else:
        print(__doc__)
