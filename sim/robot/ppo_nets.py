# SPDX-License-Identifier: MIT
"""Canonical Torch PPO network definitions used by every Warp workflow."""

from dataclasses import dataclass

from train_mesh_warp import Actor, Critic, RunningNorm

POLICY_HIDDEN = (512, 256, 128)
VALUE_HIDDEN = (512, 256, 128)


@dataclass
class PPONetworks:
    policy_network: Actor
    value_network: Critic
    normalizer: RunningNorm


def make_ppo_networks(observation_size, action_size, **kwargs):
    policy_hidden = tuple(kwargs.get("policy_hidden_layer_sizes", POLICY_HIDDEN))
    value_hidden = tuple(kwargs.get("value_hidden_layer_sizes", VALUE_HIDDEN))
    policy = Actor(int(observation_size), int(action_size), policy_hidden)
    value = Critic(int(observation_size), value_hidden)
    return PPONetworks(policy, value, RunningNorm(int(observation_size)))


def make_inference_fn(networks):
    def factory(params=None, deterministic=True):
        del params, deterministic

        def infer(obs, key=None):
            del key
            return networks.policy_network(networks.normalizer(obs)).tanh(), {}

        return infer

    return factory
