# SPDX-License-Identifier: MIT
"""Phase 8 (code) — offline integration of `PodRunner` via a MOCK pod (exec/fetch injected),
driving a real `Curriculum` through the engine. Proves the pod path end-to-end WITHOUT a GPU:
the kernel argv is built from `Stage.flags`, the remote command carries `source env.sh` +
`TRACE_*`/`ARENA_SINK` exports (so pod events join the stream), and `{tag}_state.json` is
fetched + parsed into the `res` contract.

The LIVE real-GPU exercise is the self-play transition itself — `arena.cli pipeline --runner pod`
on the existing A100 — which advances the overarching goal (no second paid pod).

  python -m arena.pod_smoke --selftest
"""

from __future__ import annotations

import json, shlex, sys, tempfile
from pathlib import Path

from arena.runner import Pod, PodRunner          # noqa: E402
from arena.engine import RunState, drive         # noqa: E402
from arena.schedule import Curriculum            # noqa: E402


def _selftest():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import curriculum_drive as cd
    seen = {}

    def fake_exec(cmd):
        # the remote command MUST carry the env-source + trace exports (the cross-machine context)
        assert "source /root/proj/out/env.sh" in cmd, cmd
        assert "TRACE_STAGE=" in cmd and "ARENA_SINK=" in cmd, cmd
        assert "cd /root/proj/sim/robot" in cmd and "train_adversarial.py" in cmd, cmd
        toks = shlex.split(cmd)
        tag = toks[toks.index("--tag") + 1]
        cum = int(toks[toks.index("--cum-base") + 1]); steps = int(toks[toks.index("--steps") + 1])
        assert "--envs" in toks and toks[toks.index("--envs") + 1] == "8192"      # runner passed envs
        seen[tag] = {"tag": tag, "cum_step": cum + steps, "best_bench": -20.0 + 0.5 * len(seen),
                     "last_ratio": 1.3}
        return 0, "compiled; trained; saved"

    def fake_fetch(path):
        tag = Path(path).name.replace("_state.json", "")
        return json.dumps(seen.get(tag, {}))

    pod = Pod(ip="1.2.3.4", port=12345, exec_fn=fake_exec, fetch_fn=fake_fetch)
    pr = PodRunner(pod, out="/root/proj/out", lean=True, envs=8192, run_id="podtest")
    st = RunState(path=str(Path(tempfile.mkdtemp()) / "s.json"))
    drive(Curriculum(steps_per_phase=1000), pr, st)
    assert st.completed == [p["name"] for p in cd.PHASES], st.completed
    assert st.cum_step == 1000 * len(cd.PHASES)
    assert st.best_ckpt and st.best_ckpt.endswith("_best.pkl")
    print("PROVEN: PodRunner drives a real pipeline offline — kernel argv + source env.sh + "
          "TRACE/ARENA_SINK exports + remote state.json fetch/parse. Real-GPU = the self-play transition.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        import os
        os.environ.setdefault("CODESIGN_OUT", tempfile.mkdtemp())   # kernel mkdir's OUT on import
        _selftest()
    else:
        print(__doc__)
