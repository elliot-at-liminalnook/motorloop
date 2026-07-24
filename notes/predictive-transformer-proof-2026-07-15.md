<!-- SPDX-License-Identifier: MIT -->
# Temporal Transformer predictor — first RunPod proof, 2026-07-15

> **Document status:** Historical · **Audience:** Robot-learning contributors · **Last reviewed:** 2026-07-15 · **Canonical for:** First recurrent-versus-Transformer future-predictor ablation

## Verdict

The causal temporal Transformer is a viable predictor implementation and makes
the predictor/planner computation substantially faster. It is **not promoted as
the default yet**: across this short run it learned faster initially, but its
final held-out forecast was less accurate than the recurrent decoder. Diffusion
sampling was deliberately not added before this simpler architectural question
was answered.

Both implementations retain the same 50 Hz GRU actor. Only the auxiliary
future-physics decoder changes. The Transformer consumes the complete candidate
action chunk in one call, applies causal attention so a predicted state cannot
depend on later actions, and emits the complete predicted physical trajectory
in parallel.

## Experiment

The matched physics runs used an NVIDIA L40S, FP32, identical seed `20260715`,
universal-control rung 2, 128 environments, 64-frame rollouts, a 32-frame
prediction horizon, four prediction anchors, two PPO epochs, eight minibatches,
and 98,304 environment steps. This is a twelve-update first-result ablation on
the stand-and-settle task, not a ladder promotion trial.

Reproduce both runs with:

```bash
bash scripts/run_predictive_decoder_proof.sh
```

The ignored machine artifacts are in `out/predictive_decoder_proof/` and include
both checkpoints, logs, JSONL metrics, structured diagnostics, and complete
per-update statistics.

The artifacts were copied back before teardown. RunPod pod `d17omwnjm3rkev` was
deleted after the comparison; a follow-up REST lookup returned `404 pod not
found`, so this proof left no billed pod running.

## Results

The isolated predictor benchmark used width 512 and horizon 32. The training
measurement used batch 16, matching one recurrent PPO minibatch; the planner
measurement differentiated through a 16-frame candidate for 128 environments.

| Predictor-only L40S FP32 measurement | Recurrent | Transformer | Transformer change |
| --- | ---: | ---: | ---: |
| Parameters | 2,135,090 | 5,289,522 | 2.48× larger |
| Training forward + backward | 19.316 ms | 6.601 ms | 2.93× faster |
| Planner action-gradient pass | 10.018 ms | 5.034 ms | 1.99× faster |

The end-to-end run includes physics, the live GRU actor, PPO, evaluation,
diagnostics, and checkpointing, so the isolated speedup is naturally diluted.

| Matched 98,304-step result | Recurrent | Transformer | Interpretation |
| --- | ---: | ---: | --- |
| Wall time at final evaluation | 108.051 s | 104.395 s | Transformer 3.4% faster |
| Final cumulative throughput | 909.8 step/s | 941.7 step/s | Transformer 3.5% higher |
| Final optimization time/update | 4.687 s | 3.759 s | Transformer 19.8% lower |
| Final total update time | 5.975 s | 5.045 s | Transformer 15.6% lower |
| Mean training prediction loss | 0.12851 | 0.11804 | Transformer 8.1% lower |
| Mean held-out calibration error | 0.15993 | 0.15676 | Transformer 2.0% lower |
| Final training prediction loss | 0.09424 | 0.10639 | Transformer 12.9% worse |
| Final held-out calibration error | 0.13032 | 0.15227 | Transformer 16.8% worse |
| Final deterministic reward | 8.24358 | 8.24228 | Effectively equal |
| Fall rate | 0 | 0 | Equal |
| Save/reload replay | pass | pass | Both valid |

Lower prediction loss and calibration error are better. The per-update curves
resolve the apparently conflicting mean and final rows: the Transformer began
better (`0.2491` versus `0.3620` calibration on update one), reached `0.1283` on
update five, then drifted to about `0.152`. The recurrent model improved more
slowly but settled near `0.130`. Thus the Transformer demonstrated faster early
learning and faster computation, but not a stable accuracy win.

Planner authority became nonzero only after ten held-out observations. Guided
versus unguided outcomes were numerically indistinguishable on this static task,
as expected: stand-and-settle supplies almost no meaningful trajectory-choice
problem. Both runs had zero falls and nearly identical reward, so there is no
evidence here that either planner improves locomotion or combat.

## Decision and next falsification

Keep `--prediction-decoder transformer` available as an experimental option;
retain `recurrent` as the default. Before adding diffusion, test the Transformer
on diverse locomotion and commanded-leg interaction rollouts and prevent the
predictor from continuing to update when held-out calibration degrades. A fair
next ablation should also separate the predictor optimizer from PPO so the
larger Transformer is not forced to share the policy learning-rate schedule.

This run proves implementation correctness and a real compute advantage. It
does not prove long-horizon skill quality, diffusion planning, hardware
performance, or replacement of the accepted ladder controller.
