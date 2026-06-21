<!-- SPDX-License-Identifier: MIT -->
# GPU pod setup — exact reproducible recipe (RunPod 4090)

The pod disk is **ephemeral** — terminating destroys `/root/proj` and all checkpoints. This
is the from-scratch recipe to rebuild the remote MJX co-design env, plus how to restore the
artifacts we pulled down. Companion to `notes/gpu-runbook.md` (what to run) and the
`motorloop-runpod-gpu` memory (API/auth). Verified on this session's pod (jax 0.6.2, brax
0.14.1, mujoco 3.9.0, Python 3.10, CUDA 12, driver 550, RTX 4090).

## 0. Provision — use the REST API (`rest.runpod.io/v1`), NOT GraphQL
**The GraphQL `podFindAndDeployOnDemand` mutation is unreliable/broken (2026-06): it 500s with
`INTERNAL_SERVER_ERROR` even when the web console shows the GPU available and the balance is fine.
Use the REST API — it's what the console uses and it works.** Also: the old
`runpod/pytorch:2.1.0-py3.10-cuda11.8.0` image is **deprecated** (silent `INTERNAL_SERVER_ERROR`
or "not found"); the current image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
(Ubuntu 24.04 / **Python 3.12** / CUDA 12.8).
```bash
KEY=$(tr -d '\n' < ~/RUNPOD_API_KEY); PUB=$(cat ~/.ssh/runpod_ed25519.pub)
IMG="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
python3 - "$KEY" "$PUB" "$IMG" <<'PY'  # create the pod (gpuTypeIds = array, env = object)
import json,sys,urllib.request
key,pub,img=sys.argv[1:4]
body=json.dumps({"cloudType":"SECURE","gpuTypeIds":["NVIDIA A100 80GB PCIe"],"gpuCount":1,
  "imageName":img,"containerDiskInGb":30,"volumeInGb":0,"ports":["22/tcp"],"env":{"PUBLIC_KEY":pub}}).encode()
r=urllib.request.Request("https://rest.runpod.io/v1/pods",body,{"Authorization":"Bearer "+key,"Content-Type":"application/json"})
print(urllib.request.urlopen(r).read().decode())   # -> {"id": "...", "costPerHr":..., "desiredStatus":"RUNNING"}
PY
# poll for the public 22/tcp ip:port via REST (publicIp + portMappings["22"]):
curl -s -H "Authorization: Bearer $KEY" https://rest.runpod.io/v1/pods/<POD_ID> \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('publicIp'),(d.get('portMappings') or {}).get('22'))"
```
GPU choice (REST `gpuTypeIds`, exact ids): `"NVIDIA A100 80GB PCIe"` ($1.39/hr, used here),
`"NVIDIA A100-SXM4-80GB"`, `"NVIDIA L40S"`, `"NVIDIA GeForce RTX 4090"`. If a type 500s/"no
resources", it's capacity — sweep types/clouds (`SECURE`/`COMMUNITY`) until one deploys. Check
balance/limit with GraphQL `query{myself{clientBalance spendLimit}}` (that query still works).

## 1. SSH helper
```bash
echo "<IP> <PORT>" > /tmp/runpod_ssh        # from the poll above
cat > /tmp/rp.sh <<'EOF'
#!/bin/bash
read IP PORT < /tmp/runpod_ssh
ssh -i ~/.ssh/runpod_ed25519 -p "$PORT" -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o ConnectTimeout=20 \
  -o LogLevel=ERROR root@"$IP" "$@"
EOF
chmod +x /tmp/rp.sh
/tmp/rp.sh 'nvidia-smi --query-gpu=name --format=csv,noheader'   # smoke
```
Use the **dedicated** `~/.ssh/runpod_ed25519` key (NOT the 1Password key — it pops biometric
prompts and blocks non-interactive ssh).

## 2. Ship code + restore artifacts (re-needed every fresh pod — disk is ephemeral)
```bash
cd <repo>
tar czf - sim/robot sim/tests/motors.py | /tmp/rp.sh 'mkdir -p /root/proj && tar xzf - -C /root/proj'
# restore the pulled checkpoints so warm-starts work (optional but saves re-training the locomotor):
tar czf - -C sim/build/gpu out | /tmp/rp.sh 'tar xzf - -C /root/proj'
```
Pulled artifacts live in `sim/build/gpu/out/` locally (universal_ckpt.pkl = the locomotor to
warm-start from; baseline/adv/f2/fspeed_ckpt.pkl; fight_metrics.jsonl; all validate_*.log).

## 3. Install — ONE command (idempotent; bakes in every fix)
```bash
/tmp/rp.sh 'bash /root/proj/sim/robot/setup_pod.sh'   # pinned install + writes out/env.sh + smoke test
# expect tail: "devices: [CudaDevice(id=0)] | brax 0.14.1" then "SETUP_OK"
```
`setup_pod.sh` installs the **pinned** `requirements-gpu.txt` (jax/jaxlib 0.6.2, brax 0.14.1,
mujoco/mujoco-mjx 3.9.0, flax 0.10.7, optax 0.2.8, numpy 2.2.6 — verified this session; unpinned
brax/jax break the saved checkpoints), creates the JAX compile cache, and writes
`out/env.sh` with the mandatory exports. Python 3.10 → `tomli` backports `tomllib`
(`gen_robot_mjcf` already falls back).

## 4. Run — source the generated env first (hard-won exports), tiny before scale
```bash
# every run starts by sourcing the env setup_pod.sh wrote (MUJOCO_GL="", PREALLOCATE=false,
# CODESIGN_OUT, JAX_COMPILATION_CACHE_DIR):
/tmp/rp.sh 'source /root/proj/out/env.sh && cd /root/proj/sim/robot && bash validate_gpu.sh'  # leak-test tiny first (E2E-first)
```
For a **long** run, launch detached and poll by sentinel (never hold an ssh blocking-wait, and
never chain the `nohup` launch with more commands in the same ssh call — that hangs ssh):
```bash
/tmp/rp.sh 'source /root/proj/out/env.sh && cd /root/proj/sim/robot && \
  nohup sh -c "timeout 1800 python3 -u combat_rank.py ... > /root/proj/out/run.log 2>&1; echo DONE_SENTINEL" \
  > /root/proj/out/launch.log 2>&1 & echo pid $!'
# then in a SEPARATE call, poll for the sentinel (or the container pid disappearing):
/tmp/rp.sh 'grep -q DONE_SENTINEL /root/proj/out/launch.log && echo done || echo running'
```

## 5. Gotchas (each cost real time this session)
- **`cuSolver internal error` = OOM/contention, not a bug.** A dead JAX process keeps ~17 GB
  (XLA preallocates 75%). Fix: `XLA_PYTHON_CLIENT_PREALLOCATE=false`; one GPU ⇒ run stages
  **strictly sequential** (overlapping JAX procs collide).
- **Host vs container PIDs.** `nvidia-smi` shows HOST pids; `ps`/`kill` inside the pod use
  CONTAINER pids. To kill a training run: `kill $(pgrep -f train_adversarial)` (container pid),
  not the nvidia-smi pid. To free a wedged GPU: kill the container pid, then check
  `nvidia-smi --query-compute-apps=pid` clears.
- **Avoid `pkill -f python3` over SSH** — it can kill the ssh session (exit 255). Kill specific
  container pids instead.
- **Compile cost ~150 s** per fresh graph; the two-robot fight scene is heavy. `setup_pod.sh`
  sets `JAX_COMPILATION_CACHE_DIR=$CODESIGN_OUT/.jax_cache` (in `out/env.sh`) to skip recompiles
  across runs.
- **Harmless startup noise — do NOT chase these:** `Failed to import warp: No module named 'warp'`
  (mujoco-mjx 3.9 probing its optional warp backend; we use the JAX backend) and
  `RuntimeWarning: overflow encountered in cast` from `jax/_src/interpreters/xla.py` (a benign
  dtype cast of a design vector). Both print on every run; neither affects results.
- **Long remote jobs: launch detached + poll, never block.** Run heavy stages under
  `nohup sh -c "timeout <s> python3 -u … > log 2>&1; echo SENTINEL" &` and poll for the SENTINEL
  in a *separate* ssh call. Chaining the `nohup` launch with more commands in one ssh invocation
  hangs the ssh (it waits on the backgrounded child's fds); a foreground `timeout` is the hard
  wall-clock cap so a wedged compile can't bill forever. Poll by **container pid** (`ps -p <pid>`),
  not the nvidia-smi (host) pid.
- **`combat_rank.py` env must match the fighter ckpt's training contacts.** The contact-forced
  fighter trains with `--lean-contacts` (`self_collision=False`); `combat_rank.py` therefore
  builds `AdversarialEnv(self_collision=False[, reality_gap=True])`. A mismatch silently changes
  the collision set (595→99 pairs) and the obs/dynamics the policy sees → garbage rankings.
- **Throughput (measured, this body):** single-body MJX ~35k env-steps/s @ batch 16384;
  two-robot fight scene ~6.8–7.8k @ batch 8192 (contact-bound). Lean contacts
  (`build_match(self_collision=False)`) ≈1.15×; saturation (2048→8192 envs) ≈6×.

## 6. Stop billing
```bash
KEY=$(tr -d '\n' < ~/RUNPOD_API_KEY)
# TERMINATE (disk is ephemeral anyway — everything needed is pulled to sim/build/gpu/out/):
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' -X POST \
  https://api.runpod.io/graphql -d '{"query":"mutation{podTerminate(input:{podId:\"<POD_ID>\"})}"}'
```
Cost discipline: write code locally (free), spin up only for GPU bursts, terminate when idle.
4090 ≈ $0.34–0.69/hr.
