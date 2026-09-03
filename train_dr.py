"""
Stable-Baselines3 PPO training script for the 2-wheel balance bot, with domain
randomization (DR). Mirrors the phases and hyperparameters used in
train_with_ppo_dr.ipynb (custom from-scratch PPO trainer), but runs them through
Stable-Baselines3 the same way train.py does.

Phases:
  1. Balance only
  2. Penalize position and rotation (yaw)
  3. Observation noise + action delay
  4. Motor noise + random pushes
  5. Mass + friction randomization
  6. Motor gain randomization
  7. Random axle torques (simulated tire ridges)
"""

import time
from pathlib import Path
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from envs.balance_bot_env_dr import BalanceBotEnv, DomainRandomConfig

# -----------------------------------------------------------------------------
# Configuration & Variables (matches PPOConfig in train_with_ppo_dr.ipynb)
# -----------------------------------------------------------------------------
MJCF_PATH = Path(__file__).parent / "robot" / "bala2-fire-simplified.xml"
SEED = 42
NUM_ENVS = 8                # Notebook uses 8 parallel envs (only env[0] renders)
STEPS_PER_ENV = 500_000
TOTAL_TIMESTEPS = NUM_ENVS * STEPS_PER_ENV  # 4,000,000 steps per phase

N_STEPS = 2048               # num_steps
NUM_MINIBATCHES = 32
BATCH_SIZE = (NUM_ENVS * N_STEPS) // NUM_MINIBATCHES  # 512
UPDATE_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
VALUE_CLIP = 1.0             # maps to SB3's clip_range_vf
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_INTERVAL_ITERS = 50  # "save every 50 iterations", as in the notebook

POLICY_KWARGS = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

# Root run directory for this whole DR curriculum (one subfolder per phase)
RUN_NAME = f"BalanceBot-v0__balance-bot-ppo-dr__{SEED}__{int(time.time())}"
BASE_LOG_DIR = Path(f"runs/{RUN_NAME}")
BASE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule (matches anneal_lr=True in PPOConfig)."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


# -----------------------------------------------------------------------------
# Environment Helpers
# -----------------------------------------------------------------------------
def make_balance_bot_env(render: bool = False, **kwargs):
    """Factory function for creating an individual environment instance."""
    env = BalanceBotEnv(
        mjcf_path=MJCF_PATH,
        render_mode="human" if render else None,
        **kwargs
    )
    return gym.wrappers.RecordEpisodeStatistics(env)


def make_vector_env(num_envs: int, render: bool = False, **kwargs) -> VecEnv:
    """
    Creates a vectorized environment.

    If render=True, ALL environments get render_mode="human" because
    Stable-Baselines3 requires matching render modes. Only env[0] will
    actually have render() called (see MujocoRenderCallback), so only one
    MuJoCo viewer window opens.
    """
    env_fns = [
        (lambda kw=kwargs, render=render: make_balance_bot_env(render=render, **kw))
        for _ in range(num_envs)
    ]
    return DummyVecEnv(env_fns)


def set_env_attrs(vec_env: VecEnv, **attrs):
    """
    Update attributes directly on the underlying BalanceBotEnv instances
    (unwrapping RecordEpisodeStatistics). Mirrors how the notebook mutates
    reward coefficients / the domain_rand config in place between phases,
    e.g. `env_stat_wrapper.env.dr = dr` or `env.position_penalty_coef = 0.001`.
    """
    for env_wrapper in vec_env.envs:
        raw_env = env_wrapper.env
        for key, value in attrs.items():
            setattr(raw_env, key, value)


# -----------------------------------------------------------------------------
# MuJoCo Rendering Callback
# -----------------------------------------------------------------------------
class MujocoRenderCallback(BaseCallback):
    """
    Updates the MuJoCo viewer for environment 0 during SB3 training.

    All environments have render_mode="human" to satisfy SB3's requirement
    that vectorized environments have matching render modes. However, we
    call render() only on environment 0, so only one viewer window opens.
    """

    def __init__(self, render_every=1, verbose=0):
        super().__init__(verbose)
        self.render_every = render_every
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        if self.step_count % self.render_every != 0:
            return True

        # DummyVecEnv.envs contains the individual environments.
        # Your environment is wrapped by RecordEpisodeStatistics.
        env_wrapper = self.training_env.envs[0]

        # Unwrap: RecordEpisodeStatistics -> BalanceBotEnv
        raw_env = env_wrapper.env

        # This creates/updates only one MuJoCo viewer.
        raw_env.render()

        return True


# -----------------------------------------------------------------------------
# Model / callback helpers
# -----------------------------------------------------------------------------
def build_model(env: VecEnv, log_dir: Path) -> PPO:
    """Build a fresh PPO model with hyperparameters matching PPOConfig."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(3e-4),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=UPDATE_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_COEF,
        clip_range_vf=VALUE_CLIP,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=str(log_dir),
        seed=SEED,
        verbose=1,
    )


def make_callbacks(log_dir: Path, eval_env: VecEnv):
    """Checkpoint / eval / render callbacks for a single phase."""
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL_ITERS * N_STEPS,  # every 50 iterations
        save_path=str(log_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir),
        log_path=str(log_dir),
        eval_freq=CHECKPOINT_INTERVAL_ITERS * N_STEPS,
        deterministic=True,
        render=False,
    )
    render_callback = MujocoRenderCallback(render_every=1)
    return [checkpoint_callback, eval_callback, render_callback]


def load_best_if_available(model: PPO, log_dir: Path, env: VecEnv) -> PPO:
    """
    Mirrors the notebook's pattern of reloading the best checkpoint from a
    phase before continuing training in the next phase:
        result.agent.load_state_dict(torch.load(result.best_model_path))
    """
    best_path = Path(log_dir) / "best_model.zip"
    if best_path.exists():
        print(f"Loading best model from {best_path}")
        model = PPO.load(best_path, env=env)
    else:
        print("No best model found for this phase; continuing with final weights.")
    return model


def run_phase(
    phase_name: str,
    model,
    train_env: VecEnv,
    eval_env: VecEnv,
    total_timesteps: int = TOTAL_TIMESTEPS,
    first_phase: bool = False,
) -> PPO:
    """Run one curriculum phase: train, save, evaluate, and reload the best model."""
    print("\n" + "=" * 60)
    print(phase_name)
    print("=" * 60)

    log_dir = BASE_LOG_DIR / phase_name
    log_dir.mkdir(parents=True, exist_ok=True)

    if first_phase:
        model = build_model(train_env, log_dir)
    else:
        # Continue training the existing agent on the (updated) environments
        model.set_env(train_env)
        model.tensorboard_log = str(log_dir)
        # Reset the linear LR schedule so each phase anneals fresh, same as
        # the notebook re-running train() with anneal_lr=True per phase.
        model.learning_rate = linear_schedule(3e-4)

    callbacks = make_callbacks(log_dir, eval_env)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=phase_name,
        reset_num_timesteps=first_phase,
    )

    model.save(log_dir / f"{phase_name}_final")

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=3, deterministic=True
    )
    print(f"{phase_name} Mean Return: {mean_reward:.2f} +/- {std_reward:.2f}")

    model = load_best_if_available(model, log_dir, train_env)

    return model


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    print(f"Starting SB3 DR Run: {RUN_NAME}")
    print(f"Logs: {BASE_LOG_DIR}")

    start_time = time.time()

    # =========================================================================
    # PHASE 1: Balance only
    # =========================================================================
    train_env = make_vector_env(
        NUM_ENVS,
        render=True,
        pitch_penalty_coef=0.5,
        action_penalty_coef=0.01,
        position_penalty_coef=0.0,
        yaw_penalty_coef=0.0,
    )
    train_env.seed(SEED)

    # Separate environment used ONLY for evaluation, so eval resets/episodes
    # don't interfere with the training rollouts.
    eval_env = make_vector_env(
        1,
        render=False,
        pitch_penalty_coef=0.5,
        action_penalty_coef=0.01,
        position_penalty_coef=0.0,
        yaw_penalty_coef=0.0,
    )
    eval_env.seed(SEED + 1)

    model = run_phase("balance-bot-phase-1", None, train_env, eval_env, first_phase=True)

    # =========================================================================
    # PHASE 2: Penalize position and rotation
    # =========================================================================
    set_env_attrs(train_env, position_penalty_coef=0.001, yaw_penalty_coef=0.1)
    set_env_attrs(eval_env, position_penalty_coef=0.001, yaw_penalty_coef=0.1)

    model = run_phase("balance-bot-phase-2", model, train_env, eval_env)

    # =========================================================================
    # PHASE 3: Observation noise and action delay
    # =========================================================================
    dr = DomainRandomConfig(
        pitch_noise_std_dev=0.01,       # Inject noise into pitch observation
        pitch_rate_noise_std_dev=0.01,  # Inject noise into pitch rate observation
        wheel_vel_noise_std_dev=0.1,    # Inject noise into wheel velocity observation
        action_delay_steps=1,           # 1 step (5ms) delay
        action_delay_random=True,       # Vary 0-1 steps each episode
    )
    set_env_attrs(train_env, dr=dr)
    set_env_attrs(eval_env, dr=dr)

    model = run_phase("balance-bot-phase-3", model, train_env, eval_env)

    # =========================================================================
    # PHASE 4: Motor noise and random pushes
    # =========================================================================
    dr.motor_noise_scale = 0.02   # +/- 2% motor noise
    dr.push_prob = 0.005          # 0.5% chance of push per step
    dr.push_force_max_n = 0.3     # Gentle pushes
    # dr is the same shared object already referenced by the envs; reassigning
    # here just mirrors the notebook's explicit per-phase update pattern.
    set_env_attrs(train_env, dr=dr)
    set_env_attrs(eval_env, dr=dr)

    model = run_phase("balance-bot-phase-4", model, train_env, eval_env)

    # =========================================================================
    # PHASE 5: Mass and friction randomization
    # =========================================================================
    dr.mass_scale_range = (0.8, 1.2)       # +/- 20% mass variation
    dr.friction_scale_range = (0.5, 1.5)   # 50-150% friction variation
    set_env_attrs(train_env, dr=dr)
    set_env_attrs(eval_env, dr=dr)

    model = run_phase("balance-bot-phase-5", model, train_env, eval_env)

    # =========================================================================
    # PHASE 6: Motor gain randomization
    # =========================================================================
    dr.motor_gain_range = (0.6, 1.0)   # 60-100% motor torque per episode
    set_env_attrs(train_env, dr=dr)
    set_env_attrs(eval_env, dr=dr)

    model = run_phase("balance-bot-phase-6", model, train_env, eval_env)

    # =========================================================================
    # PHASE 7: Random axle torques (simulate tire ridges)
    # =========================================================================
    dr.ridge_prob = 0.05               # 5% chance of tire "ridge" per step
    dr.ridge_torque_max_nm = 0.005     # Slight torque applied to the axles
    set_env_attrs(train_env, dr=dr)
    set_env_attrs(eval_env, dr=dr)

    model = run_phase("balance-bot-phase-7", model, train_env, eval_env)

    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")

    # =========================================================================
    # Wrap up
    # =========================================================================
    final_model_path = BASE_LOG_DIR / "balance-bot-ppo-dr_final"
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    train_env.close()
    eval_env.close()
    print("Training and evaluation environments closed successfully.")


if __name__ == "__main__":
    main()