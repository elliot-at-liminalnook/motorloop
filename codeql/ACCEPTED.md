<!-- SPDX-License-Identifier: MIT -->
# Accepted CodeQL findings (baseline)

Findings reviewed and deliberately kept, with rationale. A rescan reporting
ONLY these locations for these rules is considered clean. Re-review an entry
whenever its file is substantially reworked.

| rule | location | rationale |
| --- | --- | --- |
| py/call-to-non-callable | warp_dataset/warp_eval/warp_search `policy(obs)` call sites | False positive: `load_policy` returns a closure-defined `LoadedPolicy` with `__call__`; CodeQL's points-to cannot see it. |
| py/multiple-definition | `warplayer/contacts.py` (`y`) | Warp kernels require pre-declaring a variable before conditional assignment; the "dead" initial store is the type anchor. |
| py/overwritten-inherited-attribute | `ladder_warp_env.py` (`_gait`, `model_hash`), `leg_attack_warp_env.py` (`obs_dim`) | Deliberate subclass overrides, each with an in-code comment: the ladder discards the hidden base gait teacher by contract; adapters re-derive identity hashes and widths. |
| py/init-calls-subclass | `walker_warp_env.py`, `combat_warp_env.py`, `leg_attack_warp_env.py` (`reset()` in `__init__`) | Construction must run one warm physics step before CUDA graph capture; subclass readiness is handled by the `_*_ready` guard-flag convention. Restructuring init order risks the capture contract for zero behavioral gain. |
| py/import-and-import-from | `test_training_ladder.py` (`training_ladder`), `test_walker_warp.py` (`walker_warp_env`) | The module object is required for attribute monkeypatching (`LEGACY_WALK_TEACHER`, `CAT_ON`); the from-imports keep the many test references readable. |
| py/unused-global-variable | `arena/kernel_emit.py` (`_BUILT`) | False positive: read via `global` inside `_tracer()` before first write; CodeQL's global-statement analysis misses the module-level initializer's read. |
| motorloop/checkpoint-load-without-contract | `diagnose_ladder_prior.py`, `render_walker_video.py` | Read-only diagnostic/rendering tools reconstructing the checkpoint's OWN declared architecture; they interpret nothing under a foreign contract. |
| motorloop/checkpoint-load-without-contract | `train_adversarial.py` (`load_policy`, `warm_start`) | Legacy compatibility shims over the trainer format; their callers (sweep/benchmark tools) evaluate the checkpoint's own env family. |
| motorloop/torch-load-unsafe-weights-only | (none) | No accepted instances: every load site now uses `weights_only=True`. New hits are regressions. |
| py/polluting-import | `commanded_env.py`, `mesh_commanded_env.py` | Documented compatibility re-export shims (`# noqa: F403` inline); enumerating the legacy spec surface would freeze it harder, not less. |
