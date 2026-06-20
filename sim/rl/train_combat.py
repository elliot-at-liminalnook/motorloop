# SPDX-License-Identifier: MIT
"""Train the combat-dodge quadruped (notes/rl-combat-dodge-checklist.md §4).

PPO + VecNormalize with a difficulty curriculum: balance-only -> a slow pursuer ->
a fast one with a bigger bite. The pursuer is a kinematic mocap chaser, so a
falling robot never learns to dodge - it learns to stand first, then to keep its
legs clear of the strike band / stand off / leap.
  python train_combat.py --steps 2500000 --n-envs 16 --max-difficulty 0.6 --weapon spinner

Same heap-order discipline as train_dodge.py: the MAIN process never loads mujoco
(combat_env imports it); only the SB3 stack is imported here and mujoco loads
lazily inside the spawned env workers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "sim" / "build" / "rl"


def _make_env(motor="db42s03", weapon="spinner", hop_reward=False, lethal=True):
    from combat_env import make_combat_env   # lazy: mujoco loads in the worker only
    return make_combat_env(motor=motor, weapon=weapon, difficulty=0.0,
                           hop_reward=hop_reward, lethal=lethal)


class Curriculum(BaseCallback):
    """Ramp pursuer difficulty 0 -> max over the first `ramp` fraction."""
    def __init__(self, total, max_d, ramp=0.6):
        super().__init__()
        self.total, self.max_d, self.ramp = total, max_d, ramp

    def _on_rollout_start(self):
        frac = self.num_timesteps / (self.total * self.ramp)
        self.training_env.env_method("set_difficulty", min(self.max_d, frac * self.max_d))

    def _on_step(self):
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_500_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--max-difficulty", type=float, default=0.6)
    ap.add_argument("--weapon", default="spinner")
    ap.add_argument("--motor", default="db42s03")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="combat")
    ap.add_argument("--init-model", default=None,
                    help="warm-start from a saved policy (two-phase: stand then dodge)")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--hop-reward", action="store_true",
                    help="stage B: reward lifting feet above the strike band (no adversary)")
    ap.add_argument("--no-lethal", action="store_true",
                    help="stage H: strikes don't terminate (learn dodge timing safely)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(_make_env, n_envs=args.n_envs, seed=args.seed,
                        env_kwargs=dict(motor=args.motor, weapon=args.weapon,
                                        hop_reward=args.hop_reward,
                                        lethal=not args.no_lethal),
                        vec_env_cls=SubprocVecEnv,
                        vec_env_kwargs=dict(start_method="spawn"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if args.init_model:
        vp = args.init_model.replace("ppo_", "vecnorm_").replace(".zip", ".pkl")
        if Path(vp).is_file():
            venv = VecNormalize.load(vp, venv); venv.training = True
        model = PPO.load(args.init_model, env=venv, ent_coef=args.ent_coef,
                         tensorboard_log=str(OUT / "tb"))
        print(f"warm-started from {args.init_model}")
    else:
        model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed,
                    n_steps=2048, batch_size=256, n_epochs=10, gae_lambda=0.95,
                    gamma=0.99, ent_coef=args.ent_coef, learning_rate=3e-4, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    tensorboard_log=str(OUT / "tb"))
    print(f"training {args.tag}: {args.steps:,} steps, weapon={args.weapon}, "
          f"curriculum 0->{args.max_difficulty} (motor {args.motor})")
    model.learn(total_timesteps=args.steps,
                callback=Curriculum(args.steps, args.max_difficulty))
    model.save(OUT / f"ppo_{args.tag}.zip")
    venv.save(str(OUT / f"vecnorm_{args.tag}.pkl"))
    print(f"saved {OUT / f'ppo_{args.tag}.zip'} (+ vecnorm)")


if __name__ == "__main__":
    main()
