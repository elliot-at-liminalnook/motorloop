# SPDX-License-Identifier: MIT
"""Scale-aware diagnostics for PPO training and checkpoint evaluation.

The helpers in this module are deliberately independent of a particular robot
environment.  They turn tensors already produced by PPO into JSON-serializable
measurements and keep expensive autograd probes opt-in at evaluation boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn


EPS = 1.0e-12


def _float(value: torch.Tensor | float | int) -> float:
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


@torch.no_grad()
def tensor_stats(value: torch.Tensor, *, absolute: bool = False) -> dict:
    """Compact distribution statistics, including non-finite accounting."""
    flat = value.detach().reshape(-1).to(torch.float32)
    finite = torch.isfinite(flat)
    valid = flat[finite]
    out = {
        "count": int(flat.numel()),
        "nonfinite_count": int((~finite).sum()),
    }
    if not valid.numel():
        return dict(out, mean=None, std=None, min=None, max=None,
                    p10=None, p50=None, p90=None, p99=None)
    measured = valid.abs() if absolute else valid
    quantiles = torch.quantile(
        measured, measured.new_tensor((0.10, 0.50, 0.90, 0.99)))
    out.update(
        mean=_float(measured.mean()),
        std=_float(measured.std(unbiased=False)),
        min=_float(measured.min()),
        max=_float(measured.max()),
        p10=_float(quantiles[0]),
        p50=_float(quantiles[1]),
        p90=_float(quantiles[2]),
        p99=_float(quantiles[3]),
    )
    return out


@torch.no_grad()
def parameter_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone trainable parameters for an actual-update measurement."""
    return {name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad}


@torch.no_grad()
def parameter_update_diagnostics(
    module: nn.Module, before: Mapping[str, torch.Tensor], *,
    include_layers: bool = True,
) -> dict:
    """Measure the parameter step Adam actually applied, not just gradients."""
    parameter_sq = update_sq = 0.0
    max_abs_update = 0.0
    nonfinite = 0
    layers = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or name not in before:
            continue
        old = before[name].to(device=parameter.device, dtype=parameter.dtype)
        delta = parameter.detach() - old
        old_norm = _float(torch.linalg.vector_norm(old))
        update_norm = _float(torch.linalg.vector_norm(delta))
        relative = update_norm / max(old_norm, EPS)
        layer_nonfinite = int((~torch.isfinite(parameter)).sum())
        nonfinite += layer_nonfinite
        layer_max = _float(delta.abs().max()) if delta.numel() else 0.0
        max_abs_update = max(max_abs_update, layer_max)
        parameter_sq += old_norm * old_norm
        update_sq += update_norm * update_norm
        if include_layers:
            layers[name] = {
                "parameter_norm_before": old_norm,
                "update_norm": update_norm,
                "relative_update": relative,
                "max_abs_update": layer_max,
                "nonfinite_count": layer_nonfinite,
            }
    parameter_norm = math.sqrt(parameter_sq)
    update_norm = math.sqrt(update_sq)
    ordered = sorted(
        ((values["relative_update"], name) for name, values in layers.items()),
        reverse=True)
    return {
        "parameter_norm_before": parameter_norm,
        "update_norm": update_norm,
        "relative_update": update_norm / max(parameter_norm, EPS),
        "max_abs_update": max_abs_update,
        "nonfinite_parameter_count": nonfinite,
        "largest_relative_updates": [
            {"name": name, **layers[name]} for _, name in ordered[:8]
        ],
        "layers": layers,
    }


@torch.no_grad()
def optimizer_diagnostics(optimizer: torch.optim.Optimizer) -> dict:
    """Expose Adam state initialization, finiteness, scale, and effective step."""
    groups = []
    total_parameters = total_states = total_nonfinite = 0
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = [parameter for parameter in group["params"]
                      if parameter.requires_grad]
        total_parameters += len(parameters)
        states = [optimizer.state.get(parameter, {}) for parameter in parameters]
        initialized = [state for state in states
                       if "exp_avg" in state and "exp_avg_sq" in state]
        total_states += len(initialized)
        steps, first_moments, second_moments, effective_steps = [], [], [], []
        group_nonfinite = 0
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        learning_rate = float(group.get("lr", 0.0))
        epsilon = float(group.get("eps", 1.0e-8))
        for state in initialized:
            raw_step = state.get("step", 0)
            step = int(raw_step.item() if isinstance(raw_step, torch.Tensor)
                       else raw_step)
            steps.append(step)
            first = state["exp_avg"].detach().reshape(-1).to(torch.float32)
            second = state["exp_avg_sq"].detach().reshape(-1).to(torch.float32)
            group_nonfinite += int((~torch.isfinite(first)).sum()
                                   + (~torch.isfinite(second)).sum())
            first_moments.append(first.abs())
            second_moments.append(second.clamp_min(0.0).sqrt())
            if step > 0:
                first_hat = first / max(1.0 - beta1 ** step, EPS)
                second_hat = second / max(1.0 - beta2 ** step, EPS)
                effective_steps.append(
                    (learning_rate * first_hat
                     / (second_hat.clamp_min(0.0).sqrt() + epsilon)).abs())
        total_nonfinite += group_nonfinite
        groups.append({
            "index": group_index,
            "learning_rate": learning_rate,
            "parameter_tensors": len(parameters),
            "initialized_state_tensors": len(initialized),
            "state_fraction": len(initialized) / max(len(parameters), 1),
            "step_min": min(steps) if steps else 0,
            "step_max": max(steps) if steps else 0,
            "step_desynchronized": bool(steps and min(steps) != max(steps)),
            "first_moment_abs": tensor_stats(torch.cat(first_moments))
                if first_moments else None,
            "second_moment_sqrt": tensor_stats(torch.cat(second_moments))
                if second_moments else None,
            "predicted_effective_step_abs": tensor_stats(torch.cat(effective_steps))
                if effective_steps else None,
            "nonfinite_state_count": group_nonfinite,
        })
    return {
        "parameter_tensors": total_parameters,
        "initialized_state_tensors": total_states,
        "state_fraction": total_states / max(total_parameters, 1),
        "nonfinite_state_count": total_nonfinite,
        "groups": groups,
    }


def _list_stats(values: Iterable[float]) -> dict:
    values = list(float(value) for value in values)
    if not values:
        return {"count": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def gradient_clip_diagnostics(actor_norms: Iterable[float],
                              critic_norms: Iterable[float],
                              max_norm: float = 1.0) -> dict:
    """Summarize independent clip pressure and applied scale factors."""
    actor_norms, critic_norms = list(actor_norms), list(critic_norms)

    def one(values):
        norms = _list_stats(values)
        scales = [min(1.0, max_norm / max(float(value), EPS)) for value in values]
        return {
            "norm": norms,
            "scale": _list_stats(scales),
            "clipped_fraction": (sum(float(value) > max_norm for value in values)
                                 / max(len(values), 1)),
        }

    return {"max_norm": float(max_norm), "actor": one(actor_norms),
            "critic": one(critic_norms)}


@torch.no_grad()
def policy_trust_region_diagnostics(
    actor: nn.Module,
    observations: torch.Tensor,
    sampled_pre_tanh: torch.Tensor,
    old_logp: torch.Tensor,
    logp_fn,
    clip: float,
) -> dict:
    """Whole-rollout PPO trust-region and action-saturation diagnostics."""
    mean = actor(observations)
    new_logp = logp_fn(sampled_pre_tanh, mean, actor.log_std)
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    ratio_sum = ratio.sum()
    ess = ratio_sum.square() / ratio.square().sum().clamp_min(EPS)
    deterministic_action = torch.tanh(mean)
    sampled_action = torch.tanh(sampled_pre_tanh)
    return {
        "approx_kl": _float(((ratio - 1.0) - log_ratio).mean()),
        "sample_log_kl": _float((-log_ratio).mean()),
        "clip_fraction": _float(((ratio - 1.0).abs() > clip).float().mean()),
        "importance_ratio": tensor_stats(ratio),
        "effective_sample_size": _float(ess),
        "effective_sample_fraction": _float(ess / max(ratio.numel(), 1)),
        "mean_pre_tanh": tensor_stats(mean),
        "sample_pre_tanh": tensor_stats(sampled_pre_tanh),
        "deterministic_action": tensor_stats(deterministic_action),
        "sampled_action": tensor_stats(sampled_action),
        "deterministic_saturation_fraction": _float(
            (deterministic_action.abs() > 0.95).float().mean()),
        "sampled_saturation_fraction": _float(
            (sampled_action.abs() > 0.95).float().mean()),
    }


def objective_gradient_diagnostics(
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
) -> dict:
    """Per-objective actor gradient norms and pairwise cosine similarities."""
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    norms: dict[str, float] = {}
    nonfinite: dict[str, int] = {}
    for name, loss in losses.items():
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            continue
        values = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = values
        square = loss.new_zeros(())
        bad = 0
        for value in values:
            if value is None:
                continue
            bad += int((~torch.isfinite(value)).sum())
            square = square + torch.nan_to_num(value).square().sum()
        norms[name] = _float(torch.sqrt(square))
        nonfinite[name] = bad

    cosines = {}
    names = list(gradients)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            dot = losses[left].new_zeros(())
            for left_value, right_value in zip(gradients[left], gradients[right]):
                if left_value is not None and right_value is not None:
                    dot = dot + (torch.nan_to_num(left_value)
                                 * torch.nan_to_num(right_value)).sum()
            denominator = max(norms[left] * norms[right], EPS)
            cosines[f"{left}__{right}"] = _float(dot) / denominator
    return {"norms": norms, "cosines": cosines,
            "nonfinite_gradient_counts": nonfinite}


@torch.no_grad()
def critic_calibration(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Scale, bias, calibration, and explained-variance diagnostics."""
    prediction = prediction.detach().reshape(-1).to(torch.float32)
    target = target.detach().reshape(-1).to(torch.float32)
    residual = prediction - target
    target_var = target.var(unbiased=False)
    prediction_var = prediction.var(unbiased=False)
    covariance = ((prediction - prediction.mean())
                  * (target - target.mean())).mean()
    slope = covariance / prediction_var.clamp_min(EPS)
    intercept = target.mean() - slope * prediction.mean()
    correlation = covariance / torch.sqrt(
        prediction_var.clamp_min(EPS) * target_var.clamp_min(EPS))
    rmse = torch.sqrt(residual.square().mean())
    explained = 1.0 - residual.var(unbiased=False) / target_var.clamp_min(EPS)
    return {
        "bias": _float(residual.mean()),
        "rmse": _float(rmse),
        "normalized_rmse": _float(rmse / target.std(unbiased=False).clamp_min(EPS)),
        "correlation": _float(correlation),
        "calibration_slope": _float(slope),
        "calibration_intercept": _float(intercept),
        "explained_variance": _float(explained),
        "prediction": tensor_stats(prediction),
        "target": tensor_stats(target),
        "residual": tensor_stats(residual),
    }


@torch.no_grad()
def normalization_snapshot(normalizer: nn.Module) -> dict[str, torch.Tensor]:
    return {"mean": normalizer.mean.detach().clone(),
            "var": normalizer.var.detach().clone(),
            "count": normalizer.count.detach().clone()}


@torch.no_grad()
def normalization_diagnostics(
    before: Mapping[str, torch.Tensor],
    normalizer: nn.Module,
    raw_observations: torch.Tensor,
) -> dict:
    """Running-normalizer drift and observation clipping/outlier diagnostics."""
    old_mean = before["mean"].to(raw_observations.device)
    old_var = before["var"].to(raw_observations.device)
    old_std = torch.sqrt(old_var + 1.0e-8)
    new_std = torch.sqrt(normalizer.var + 1.0e-8)
    mean_drift = (normalizer.mean - old_mean).abs() / old_std
    std_ratio = new_std / old_std
    old_z = (raw_observations - old_mean) / old_std
    new_z = (raw_observations - normalizer.mean) / new_std
    ranked = torch.argsort(mean_drift, descending=True)[:8]
    return {
        "count_before": _float(before["count"]),
        "count_after": _float(normalizer.count),
        "standardized_mean_drift": tensor_stats(mean_drift),
        "std_ratio": tensor_stats(std_ratio),
        "old_outlier_fraction": _float((old_z.abs() > normalizer.clip).float().mean()),
        "current_outlier_fraction": _float((new_z.abs() > normalizer.clip).float().mean()),
        "raw_nonfinite_count": int((~torch.isfinite(raw_observations)).sum()),
        "largest_mean_drift_dimensions": [
            {"index": int(index), "standardized_drift": _float(mean_drift[index]),
             "std_ratio": _float(std_ratio[index])}
            for index in ranked
        ],
    }


def scalar_metric_gap(training: Mapping, evaluation: Mapping,
                      keys: Iterable[str]) -> dict:
    """Signed and scale-free train/evaluation gaps for common scalar metrics."""
    out = {}
    for key in keys:
        train_value, eval_value = training.get(key), evaluation.get(key)
        if not isinstance(train_value, (int, float)) or not isinstance(
                eval_value, (int, float)):
            continue
        difference = float(eval_value) - float(train_value)
        out[key] = {
            "training": float(train_value), "evaluation": float(eval_value),
            "difference": difference,
            "relative_difference": difference / max(abs(float(train_value)), 1.0e-9),
        }
    return out


def checkpoint_replay_comparison(before: Mapping, after: Mapping,
                                 keys: Iterable[str], *,
                                 atol: float = 1.0e-6,
                                 rtol: float = 1.0e-6) -> dict:
    """Compare replay metrics with an absolute-plus-relative float tolerance.

    GPU physics reductions are not bitwise deterministic, so exact trajectory
    hashes remain useful evidence but cannot be the sole checkpoint contract.
    Scaling the tolerance prevents one last-place rounding quantum in a large
    accumulated reward from being treated like a meaningful behavior change.
    """
    differences, tolerances, tolerance_ratios, failures = {}, {}, {}, []
    for key in keys:
        before_value, after_value = before.get(key), after.get(key)
        if not isinstance(before_value, (int, float)) or not isinstance(
                after_value, (int, float)):
            continue
        difference = abs(float(after_value) - float(before_value))
        tolerance = float(atol) + float(rtol) * max(
            abs(float(before_value)), abs(float(after_value)))
        ratio = difference / max(tolerance, EPS)
        differences[key] = difference
        tolerances[key] = tolerance
        tolerance_ratios[key] = ratio
        if ratio > 1.0:
            failures.append(key)
    return {
        "pass": not failures,
        "atol": float(atol),
        "rtol": float(rtol),
        "max_abs_metric_difference": max(differences.values(), default=0.0),
        "max_tolerance_ratio": max(tolerance_ratios.values(), default=0.0),
        "metric_differences": differences,
        "metric_tolerances": tolerances,
        "metric_tolerance_ratios": tolerance_ratios,
        "failed_metrics": failures,
    }


def multi_seed_summary(results: list[Mapping], keys: Iterable[str]) -> dict:
    """Mean, uncertainty, and worst observed value across held-out seeds."""
    out = {"count": len(results), "metrics": {}}
    for key in keys:
        values = [float(row[key]) for row in results
                  if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        out["metrics"][key] = {
            "mean": mean, "std": std,
            "ci95_half_width": 1.96 * std / math.sqrt(len(values)),
            "min": min(values), "max": max(values), "values": values,
        }
    return out


def diagnostic_alerts(diagnostics: Mapping, gates: Mapping | None = None,
                      evaluation: Mapping | None = None) -> list[dict]:
    """Convert common failure signatures into concise, machine-readable alerts."""
    alerts: list[dict] = []

    def add(severity: str, code: str, message: str, value=None):
        row = {"severity": severity, "code": code, "message": message}
        if value is not None:
            row["value"] = value
        alerts.append(row)

    evaluation = evaluation or {}
    integrity = diagnostics.get("integrity", {})
    if integrity.get("nonfinite_count", 0):
        add("critical", "nonfinite", "Non-finite training data or parameters detected.",
            integrity["nonfinite_count"])
    if integrity.get("rollout_samples_dropped", 0):
        add("critical", "rollout_samples_dropped",
            "The minibatch partition silently omitted rollout samples.",
            integrity["rollout_samples_dropped"])
    optimizer = diagnostics.get("optimizer", {})
    if optimizer.get("nonfinite_state_count", 0):
        add("critical", "optimizer_state_nonfinite",
            "Adam's running moments contain non-finite values.",
            optimizer["nonfinite_state_count"])
    if any(group.get("step_desynchronized", False)
           for group in optimizer.get("groups", [])):
        add("warning", "optimizer_step_desynchronized",
            "Adam parameters disagree about how many updates they received.")
    kl_controller = diagnostics.get("kl_controller", {})
    if kl_controller.get("early_stop", False):
        add("info", "ppo_kl_epoch_stop",
            "PPO skipped remaining epochs because the measured policy move was large.",
            kl_controller.get("epochs_completed"))
    trust = diagnostics.get("trust_region", {})
    if trust.get("clip_fraction", 0.0) > 0.35:
        add("warning", "ppo_clip_pressure", "More than 35% of rollout ratios are clipped.",
            trust["clip_fraction"])
    if trust.get("effective_sample_fraction", 1.0) < 0.70:
        add("warning", "low_effective_sample_size",
            "Importance weights concentrate on too few rollout samples.",
            trust["effective_sample_fraction"])
    if trust.get("sampled_saturation_fraction", 0.0) > 0.20:
        add("warning", "policy_action_saturation",
            "More than 20% of sampled actions are near the control limits.",
            trust["sampled_saturation_fraction"])
    policy_std = diagnostics.get("policy_distribution", {}).get("std", {})
    if isinstance(policy_std.get("min"), (int, float)) and policy_std["min"] < 0.01:
        add("warning", "policy_exploration_collapsed",
            "At least one action dimension has almost no exploration noise.",
            policy_std["min"])
    if isinstance(policy_std.get("max"), (int, float)) and policy_std["max"] > 2.0:
        add("warning", "policy_exploration_excessive",
            "At least one action dimension has extremely large exploration noise.",
            policy_std["max"])
    clipping = diagnostics.get("gradient_clipping", {})
    actor_clip = clipping.get("actor", {}).get("clipped_fraction", 0.0)
    critic_clip = clipping.get("critic", {}).get("clipped_fraction", 0.0)
    if actor_clip > 0.80:
        add("warning", "actor_always_clipped", "Actor gradients clip on most minibatches.",
            actor_clip)
    if critic_clip > 0.80:
        add("warning", "critic_always_clipped", "Critic gradients clip on most minibatches.",
            critic_clip)
    updates = diagnostics.get("parameter_updates", {})
    actor_update = updates.get("actor", {})
    if actor_update.get("relative_update", 0.0) > 0.05:
        add("warning", "actor_update_too_large",
            "The policy moved by more than 5% of its parameter norm in one update.",
            actor_update["relative_update"])
    if actor_update.get("relative_update", 1.0) < 1.0e-10:
        add("warning", "actor_update_stalled",
            "The policy parameters did not measurably change.",
            actor_update.get("relative_update"))
    objectives = diagnostics.get("objective_gradients", {})
    conflicting = {name: value for name, value in objectives.get("cosines", {}).items()
                   if value < -0.75}
    if conflicting:
        name, value = min(conflicting.items(), key=lambda item: item[1])
        add("warning", "objective_gradient_conflict",
            f"Two policy objectives strongly oppose each other: {name}.", value)
    adaptive = diagnostics.get("adaptive_contracts", {})
    dual_max = float(adaptive.get("dual_max", 10.0))
    saturated_constraints = [
        row for row in adaptive.get("constraints", [])
        if row.get("dual", 0.0) >= 0.99 * dual_max
        and row.get("observed", 0.0) > row.get("target", float("inf"))
    ]
    if saturated_constraints:
        row = max(saturated_constraints,
                  key=lambda item: item["observed"] / max(item["target"], 1.0e-12))
        add("warning", "adaptive_constraint_saturated",
            f"Adaptive constraint {row['name']} is still violated at the dual ceiling.",
            row)
    saturated_competence = [
        row for row in adaptive.get("competence", [])
        if row.get("dual", 0.0) >= 0.99 * dual_max
        and row.get("observed", float("inf")) < row.get("target", 0.0)
    ]
    if saturated_competence:
        row = min(saturated_competence,
                  key=lambda item: item["observed"] / max(item["target"], 1.0e-12))
        add("warning", "adaptive_competence_saturated",
            f"Adaptive competence {row['name']} remains below target at the dual ceiling.",
            row)
    critic = diagnostics.get("critic", {}).get("after_update", {})
    if critic.get("normalized_rmse", 0.0) > 2.0:
        add("warning", "critic_scale_error",
            "Value prediction error is more than twice the natural return scale.",
            critic["normalized_rmse"])
    if critic.get("explained_variance", 0.0) < -0.10:
        add("warning", "critic_worse_than_constant",
            "The critic predicts returns worse than a constant baseline.",
            critic["explained_variance"])
    norm_drift = diagnostics.get("observation_normalization", {}).get(
        "standardized_mean_drift", {})
    if (norm_drift.get("p99") or 0.0) > 3.0:
        add("warning", "observation_distribution_shift",
            "Observation means moved by over three prior standard deviations.",
            norm_drift["p99"])
    seed_metrics = diagnostics.get("multi_seed_evaluation", {}).get("metrics", {})
    fragile = {}
    for name in ("reward", "xprogress", "track"):
        row = seed_metrics.get(name, {})
        if len(row.get("values", [])) < 2:
            continue
        relative_std = row.get("std", 0.0) / max(abs(row.get("mean", 0.0)), 0.01)
        if relative_std > 0.75:
            fragile[name] = relative_std
    if fragile:
        name, value = max(fragile.items(), key=lambda item: item[1])
        add("warning", "seed_fragility",
            f"Held-out {name} changes sharply with random seed.", value)
    gap = diagnostics.get("train_eval_gap", {})
    large_gap = {name: abs(row.get("relative_difference", 0.0))
                 for name, row in gap.items()
                 if name in ("reward", "xprogress", "track")
                 and abs(row.get("relative_difference", 0.0)) > 1.0}
    if large_gap:
        name, value = max(large_gap.items(), key=lambda item: item[1])
        add("warning", "train_eval_gap",
            f"Evaluation {name} differs greatly from the training rollout.", value)
    stochastic_gap = diagnostics.get("deterministic_stochastic_gap", {}).get("reward", {})
    if abs(stochastic_gap.get("relative_difference", 0.0)) > 1.0:
        add("warning", "stochastic_policy_gap",
            "Sampled-action reward differs greatly from deterministic policy reward.",
            stochastic_gap["relative_difference"])

    actuator = evaluation.get("actuator_diagnostics", {})
    for metric, threshold, code, message in (
        ("effort_saturated", 0.20, "actuator_effort_saturation",
         "Actuators request unavailable effort on more than 20% of samples."),
        ("command_saturated", 0.20, "actuator_command_saturation",
         "Direct actuator commands sit near their limits on more than 20% of samples."),
        ("target_clamped", 0.05, "joint_target_clamping",
         "Policy targets exceed joint limits on more than 5% of samples."),
        ("joint_limit_near", 0.10, "joint_limit_pressure",
         "Joints spend more than 10% of samples near a hard limit."),
    ):
        value = actuator.get(metric, {}).get("overall", {}).get("mean")
        if isinstance(value, (int, float)) and value > threshold:
            add("warning", code, message, value)
    simulation = evaluation.get("simulation_diagnostics", {})
    state_bad = simulation.get("state_nonfinite", {}).get("max", 0.0) or 0.0
    if state_bad > 0.0:
        add("critical", "simulation_state_nonfinite",
            "The physics state produced NaN or infinity.", state_bad)
    rows = simulation.get("constraint_rows", {}).get("p99")
    capacity = simulation.get("constraint_capacity", {}).get("p10")
    if isinstance(rows, (int, float)) and isinstance(capacity, (int, float)) \
            and capacity > 0.0 and rows / capacity > 0.90:
        add("warning", "solver_constraint_capacity",
            "Physics constraints approach the solver's allocated capacity.",
            rows / capacity)
    penetration = simulation.get("foot_penetration", {}).get("p99")
    if isinstance(penetration, (int, float)) and penetration > 0.01:
        add("warning", "contact_penetration",
            "One percent of foot contacts penetrate more than one centimetre.",
            penetration)
    duty = [evaluation.get(f"duty_{leg}") for leg in ("fl", "fr", "rl", "rr")]
    if all(isinstance(value, (int, float)) for value in duty):
        if min(duty) > 0.98:
            add("warning", "all_legs_planted",
                "Every leg stays planted, so a moving policy may have collapsed to standing.",
                min(duty))
        if max(duty) - min(duty) > 0.25:
            add("warning", "leg_duty_asymmetry",
                "One leg's contact duty differs substantially from another's.",
                max(duty) - min(duty))
    components = evaluation.get("reward_components", {})
    dominant = ([(name, row.get("absolute_mean_share", 0.0))
                 for name, row in components.items()
                 if row.get("absolute_mean_share", 0.0) > 0.85]
                if len(components) > 1 else [])
    if dominant:
        name, value = max(dominant, key=lambda item: item[1])
        add("warning", "reward_component_dominance",
            f"Reward component {name} supplies over 85% of absolute reward scale.", value)
    fall_rate = evaluation.get("termination_ledger", {}).get("fall", {}).get("rate")
    if isinstance(fall_rate, (int, float)) and fall_rate > 0.01:
        add("warning", "frequent_falls",
            "More than 1% of evaluation world-steps end in a fall.", fall_rate)
    replay = diagnostics.get("checkpoint_replay", {})
    if replay and not replay.get("pass", True):
        add("critical", "checkpoint_replay_mismatch",
            "Reloaded checkpoint does not reproduce its pre-save rollout.")
    if gates and not gates.get("all_pass", True):
        add("info", "gate_failure", f"Worst acceptance gate: {gates.get('worst_metric')}")
    return alerts
