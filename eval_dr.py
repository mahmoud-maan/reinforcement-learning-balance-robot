"""
Evaluate a trained PPO balance-bot policy (trained with domain randomization),
watching it live in the MuJoCo viewer.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from envs.balance_bot_env_dr import BalanceBotEnv, DomainRandomConfig

MJCF_PATH = Path(__file__).parent / "robot" / "bala2-fire-simplified.xml"

# Full domain randomization config, matching the final state of the DR
# curriculum (Phase 7) in train_dr.py. Used by default so the evaluation
# stresses the policy the same way the end of training did.
FULL_DR_CONFIG = DomainRandomConfig(
    pitch_noise_std_dev=0.01,
    pitch_rate_noise_std_dev=0.01,
    wheel_vel_noise_std_dev=0.1,
    action_delay_steps=1,
    action_delay_random=True,
    motor_noise_scale=0.02,
    push_prob=0.005,
    push_force_max_n=0.3,
    mass_scale_range=(0.8, 1.2),
    friction_scale_range=(0.5, 1.5),
    motor_gain_range=(0.6, 1.0),
    ridge_prob=0.05,
    ridge_torque_max_nm=0.005,
)


def evaluate(
    model_path,
    episodes=5,
    deterministic=True,
    real_time=True,
    domain_rand=FULL_DR_CONFIG,
    **env_kwargs,
):
    """
    Run a trained model for a fixed number of episodes, rendering live in the
    MuJoCo viewer.

    Args:
        model_path (str or Path): path to the saved .zip model (PPO.load)
        episodes (int): number of complete episodes to run
        deterministic (bool): True = always take the actor's mean action
            (no exploration noise) - what you want to judge "final" policy
            quality. False = sample from the Gaussian, like during training -
            useful to see how much the policy still relies on stochasticity.
        real_time (bool): pace playback to match the MJCF's physics timestep
            so it's watchable at normal speed, instead of fast-forwarding.
        domain_rand (DomainRandomConfig or None): domain randomization config
            to apply during evaluation. Defaults to FULL_DR_CONFIG (matching
            the end of the training curriculum) so returns are comparable to
            the toughest phase seen during training. Pass None to evaluate
            in a clean, noise-free environment instead.
        **env_kwargs: reward-shaping coefficients passed to BalanceBotEnv.
            Defaults below match Phase 2+ of training so the printed returns
            are directly comparable to what you saw during training.
    """
    env = BalanceBotEnv(
        mjcf_path=MJCF_PATH,
        render_mode="human",
        domain_rand=domain_rand,
        **env_kwargs,
    )

    model = PPO.load(model_path)

    episode_returns = []
    episode_lengths = []

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        ep_len = 0

        while not (terminated or truncated):
            step_start = time.time()
            # model.predict handles the single (non-vectorized) observation
            # automatically - no need to add a batch dimension yourself.
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_return += reward
            ep_len += 1

            env.render()

            if real_time:
                # Pace this step to match the MJCF's physics timestep so
                # playback looks real-time instead of fast-forward.
                slack = env.model.opt.timestep - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        outcome = "tipped over" if terminated else "reached max steps"
        print(f"Episode {ep + 1}: return={ep_return:.2f}, length={ep_len} ({outcome})")

    env.close()

    print("\n--- Summary ---")
    print(f"Mean return: {np.mean(episode_returns):.2f} (+/- {np.std(episode_returns):.2f})")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")

    return episode_returns, episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode_path_default = "/Users/mahmoudmaan/Documents/Learning/2026/bala-rl/runs/BalanceBot-v0__balance-bot-ppo-dr__42__1788108938/balance-bot-ppo-dr_final"
    parser.add_argument(
        "--model", type=str, default=mode_path_default,
        help="Path to the saved SB3 model (.zip)",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Sample actions instead of using the deterministic mean action",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Run as fast as possible instead of real-time paced",
    )
    parser.add_argument(
        "--no-dr", action="store_true",
        help="Evaluate in a clean environment with domain randomization disabled",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        real_time=not args.fast,
        domain_rand=None if args.no_dr else FULL_DR_CONFIG,
        # Match Phase 2+'s reward shaping so the printed returns are
        # meaningful/comparable to the training logs you already saw.
        pitch_penalty_coef=0.5,
        action_penalty_coef=0.01,
        position_penalty_coef=0.001,
        yaw_penalty_coef=0.1,
    )