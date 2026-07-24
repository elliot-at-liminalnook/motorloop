import math

import mujoco
import torch

from predictive_control import (
    MORPH_NUMERIC_DIM, InteractionTrajectoryTarget, MorphologyTokenEncoder,
    RecurrentTrajectoryDecoder, TemporalTransformerTrajectoryDecoder,
    TRAJECTORY_RAW_DIM, TRAJECTORY_SLICES, TRAJECTORY_TARGET_DIM,
    guided_action_sequence, model_morphology_tokens,
    stabilized_trajectory_target, trajectory_calibration_metrics,
    trajectory_prediction_loss)


def test_compiled_model_becomes_typed_masked_tokens():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="link" pos="0 0 1">
      <joint name="hinge" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" size=".02 .1" mass="2"/>
    </body></worldbody><actuator><motor joint="hinge" gear="3"/></actuator></mujoco>
    """)
    numeric, kinds, mask = model_morphology_tokens(
        model, actuator_wfree=[4.0], actuator_kp=[5.0], max_tokens=8, batch=3)
    assert numeric.shape == (3, 8, MORPH_NUMERIC_DIM)
    assert kinds.shape == mask.shape == (3, 8)
    assert mask[0].sum() == model.nbody - 1 + model.njnt + model.nu
    assert torch.equal(numeric[0], numeric[2])
    assert not mask[:, -1].any()


def _snapshot(x=0.0, yaw=0.0):
    state = torch.zeros(TRAJECTORY_RAW_DIM)
    state[0] = x
    state[2] = 0.4
    state[3] = math.cos(yaw / 2)
    state[6] = math.sin(yaw / 2)
    feet = torch.tensor([
        .2, .2, 0., .2, -.2, 0., -.2, .2, 0., -.2, -.2, 0.])
    feet[0::3] += x
    state[13:25] = feet
    state[25:29] = 1.0
    state[47] = 1.0
    state[48] = 0.4
    return state


def test_stabilized_target_is_relative_to_anchor_not_global_origin():
    anchor = _snapshot(x=10.0)
    future = _snapshot(x=10.5)
    target = stabilized_trajectory_target(anchor, future)
    assert target.shape == (TRAJECTORY_TARGET_DIM,)
    torch.testing.assert_close(target[TRAJECTORY_SLICES.root_delta],
                               torch.tensor([.5, 0., 0.]))
    shifted = stabilized_trajectory_target(_snapshot(x=-7.0), _snapshot(x=-6.5))
    torch.testing.assert_close(target, shifted)


def test_decoder_loss_and_guidance_are_differentiable():
    batch, horizon, width, act_dim = 4, 6, 32, 14
    encoder = MorphologyTokenEncoder(width, layers=1)
    decoder = RecurrentTrajectoryDecoder(width, act_dim, width)
    numeric = torch.randn(batch, 5, MORPH_NUMERIC_DIM)
    kinds = torch.ones(batch, 5, dtype=torch.long)
    mask = torch.ones(batch, 5, dtype=torch.bool)
    morphology = encoder(numeric, kinds, mask)
    context = torch.randn(batch, width)
    actions = torch.randn(horizon, batch, act_dim)
    prediction = decoder(context, morphology, actions)
    target = torch.zeros_like(prediction)
    loss, parts = trajectory_prediction_loss(prediction, target)
    loss.backward()
    assert parts.keys() == {"body_position", "body_rotation", "body_velocity",
                            "feet", "contact", "interaction", "effort", "fall"}
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    planned, costs = guided_action_sequence(
        decoder, context, morphology.detach(), actions, steps=2)
    assert planned.shape == actions.shape
    assert torch.isfinite(planned).all()
    assert {"before_total", "after_total", "gradient_rms",
            "action_delta_rms", "action_delta_max"} <= set(costs)


def test_transformer_decoder_is_parallel_causal_and_differentiable():
    batch, horizon, width, act_dim = 3, 8, 32, 14
    decoder = TemporalTransformerTrajectoryDecoder(
        width, act_dim, width, layers=1, heads=4)
    context = torch.randn(batch, width)
    morphology = torch.randn(batch, width)
    actions = torch.randn(horizon, batch, act_dim, requires_grad=True)
    prediction = decoder(context, morphology, actions)
    assert prediction.shape == (horizon, batch, TRAJECTORY_TARGET_DIM)

    changed = actions.detach().clone()
    changed[-1].add_(100.0)
    changed_prediction = decoder(context, morphology, changed)
    torch.testing.assert_close(prediction.detach()[:-1], changed_prediction[:-1])

    prediction.square().mean().backward()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    planned, costs = guided_action_sequence(
        decoder, context, morphology, actions.detach(), steps=1)
    assert planned.shape == actions.shape
    assert float(costs["gradient_rms"]) > 0.0


def test_interaction_target_guides_task_space_and_reports_effect():
    batch, horizon, width, act_dim = 3, 8, 32, 14
    decoder = RecurrentTrajectoryDecoder(width, act_dim, width)
    context = torch.randn(batch, width)
    morphology = torch.randn(batch, width)
    actions = torch.zeros(horizon, batch, act_dim)
    target = InteractionTrajectoryTarget.empty(
        horizon, batch, device=actions.device, dtype=actions.dtype, dt=0.02)
    target.root_delta[..., 0] = torch.linspace(.02, .16, horizon)[:, None]
    target.root_delta_mask[..., 0] = 1.0
    target.velocity[..., 0] = 0.4
    target.velocity_mask[..., 0] = 1.0

    planned, costs = guided_action_sequence(
        decoder, context, morphology, actions, steps=2,
        interaction_target=target)

    assert torch.isfinite(planned).all()
    assert float(costs["gradient_rms"]) > 0.0
    assert float(costs["action_delta_rms"]) > 0.0
    assert "before_task_root" in costs and "after_task_root" in costs


def test_calibration_is_dimensionless_and_penalizes_bad_forecasts():
    target = torch.zeros(5, 2, TRAJECTORY_TARGET_DIM)
    exact = trajectory_calibration_metrics(target, target)
    wrong = target.clone()
    wrong[..., TRAJECTORY_SLICES.root_delta] = 0.25
    wrong[..., TRAJECTORY_SLICES.contact] = 8.0
    bad = trajectory_calibration_metrics(wrong, target)

    assert float(exact["body_position"]) == 0.0
    assert float(bad["body_position"]) == 1.0
    assert float(bad["overall"]) > float(exact["overall"])
