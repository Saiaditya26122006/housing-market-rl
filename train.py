"""
Two-stage PPO training pipeline for the multi-agent housing market simulation.

Stage 1: Train each household agent type independently against heuristic opponents.
Stage 2: Train the Policy Maker (government) against frozen Stage 1 household policies.

All models save as SB3 .zip files via stable-baselines3 PPO.
"""

import argparse
import logging
from pathlib import Path

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import agents as agents_module
import config
from environment import HousingMarketEnv
from utils import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


HEURISTICS = {
    "displaced": agents_module.DisplacedHeuristic(),
    "renter": agents_module.RenterHeuristic(),
    "owner": agents_module.OwnerHeuristic(),
    "investor": agents_module.InvestorHeuristic(),
    "government": agents_module.GovernmentHeuristic(),
}


def _get_agent_type(agent_id: str) -> str:
    if agent_id == "government":
        return "government"
    for prefix in ("displaced", "renter", "owner", "investor"):
        if agent_id.startswith(prefix):
            return prefix
    return "displaced"


class SingleAgentWrapper(gymnasium.Env):
    """Wraps the multi-agent env to expose a single-agent Gym interface."""

    def __init__(
        self,
        env: HousingMarketEnv,
        target_agent_id: str,
        policies: dict | None = None,
    ):
        super().__init__()
        self.env = env
        self.target_agent_id = target_agent_id
        self.policies = policies or {}

        self.observation_space = env.observation_spaces[target_agent_id]
        self.action_space = env.action_spaces[target_agent_id]

    def _get_action_for(self, agent_id: str, obs: np.ndarray) -> int:
        if agent_id in self.policies:
            policy = self.policies[agent_id]
            if callable(policy):
                return policy(obs)
        agent_type = _get_agent_type(agent_id)
        if agent_type in HEURISTICS:
            return HEURISTICS[agent_type].predict(obs)
        return 0

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        obs = observations.get(self.target_agent_id, np.zeros(
            self.observation_space.shape, dtype=np.float32
        ))
        info = infos.get(self.target_agent_id, {})
        return obs, info

    def step(self, action):
        actions = {}
        for agent_id in self.env.agents:
            if agent_id == self.target_agent_id:
                actions[agent_id] = int(action)
            else:
                obs = self._last_observations.get(
                    agent_id,
                    np.zeros(
                        self.env.observation_spaces[agent_id].shape,
                        dtype=np.float32,
                    ),
                )
                actions[agent_id] = self._get_action_for(agent_id, obs)

        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self._last_observations = observations

        obs = observations.get(self.target_agent_id, np.zeros(
            self.observation_space.shape, dtype=np.float32
        ))
        reward = rewards.get(self.target_agent_id, 0.0)
        done = terminated.get(self.target_agent_id, False)
        trunc = truncated.get(self.target_agent_id, False)
        info = infos.get(self.target_agent_id, {})

        return obs, reward, done, trunc, info


class PolicyMakerWrapper(gymnasium.Env):
    """Wraps env for government training with frozen household policies."""

    def __init__(
        self,
        env: HousingMarketEnv,
        frozen_models: dict,
    ):
        super().__init__()
        self.env = env
        self.frozen_models = frozen_models
        self.target_agent_id = "government"

        self.observation_space = env.observation_spaces["government"]
        self.action_space = env.action_spaces["government"]

    def _get_household_action(self, agent_id: str, obs: np.ndarray) -> int:
        agent_type = _get_agent_type(agent_id)
        model = self.frozen_models.get(agent_type)
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        return HEURISTICS[agent_type].predict(obs)

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        obs = observations.get("government", np.zeros(
            self.observation_space.shape, dtype=np.float32
        ))
        info = infos.get("government", {})
        return obs, info

    def step(self, action):
        actions = {}
        for agent_id in self.env.agents:
            if agent_id == "government":
                actions[agent_id] = int(action)
            else:
                obs = self._last_observations.get(
                    agent_id,
                    np.zeros(
                        self.env.observation_spaces[agent_id].shape,
                        dtype=np.float32,
                    ),
                )
                actions[agent_id] = self._get_household_action(agent_id, obs)

        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        self._last_observations = observations

        obs = observations.get("government", np.zeros(
            self.observation_space.shape, dtype=np.float32
        ))
        reward = rewards.get("government", 0.0)
        done = terminated.get("government", False)
        trunc = truncated.get("government", False)
        info = infos.get("government", {})

        return obs, reward, done, trunc, info


class LoggingCallback(BaseCallback):
    """Logs mean reward every N steps and tracks learning curve data."""

    def __init__(self, agent_type: str, log_interval: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.agent_type = agent_type
        self.log_interval = log_interval
        self.timesteps_log = []
        self.rewards_log = []
        self.ep_len_log = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_interval == 0:
            if self.model.ep_info_buffer:
                mean_reward = np.mean(
                    [ep["r"] for ep in self.model.ep_info_buffer]
                )
                mean_len = np.mean(
                    [ep["l"] for ep in self.model.ep_info_buffer]
                )
                self.timesteps_log.append(self.num_timesteps)
                self.rewards_log.append(mean_reward)
                self.ep_len_log.append(mean_len)
                logger.info(
                    f"[{self.agent_type}] step={self.num_timesteps} "
                    f"mean_reward={mean_reward:.2f} ep_len={mean_len:.0f}"
                )
        return True




def _build_ppo(wrapper: gymnasium.Env, agent_type: str) -> PPO:
    t = config.TRAINING
    return PPO(
        "MlpPolicy",
        wrapper,
        learning_rate=t["ppo_learning_rate"],
        n_steps=t["ppo_n_steps"],
        batch_size=t["ppo_batch_size"],
        n_epochs=t["ppo_n_epochs"],
        policy_kwargs={"net_arch": list(t["net_arch"])},
        device=t["device"],
        verbose=0,
    )


def _representative_agent_id(env: HousingMarketEnv, agent_type: str) -> str:
    if agent_type == "government":
        return "government"
    return f"{agent_type}_0"


def train_stage1(
    preset_name: str | None = None,
    seed: int | None = None,
    callbacks_out: dict | None = None,
) -> dict:
    """Train household agents independently against heuristic opponents."""
    models_dir = Path(config.PATHS["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    household_types = ["displaced", "renter", "owner", "investor"]
    trained_models = {}

    for agent_type in household_types:
        logger.info(f"Stage 1: Training {agent_type}...")
        env = make_env(preset_name)
        if seed is not None:
            env.reset(seed=seed)

        agent_id = _representative_agent_id(env, agent_type)
        wrapper = SingleAgentWrapper(env, agent_id)

        model = _build_ppo(wrapper, agent_type)
        timesteps = config.TRAINING["stage1_timesteps"][agent_type]
        callback = LoggingCallback(agent_type)

        model.learn(total_timesteps=timesteps, callback=callback)

        save_path = config.PATHS[f"{agent_type}_policy"]
        model.save(save_path)
        logger.info(f"Stage 1: {agent_type} saved to {save_path}.zip")

        trained_models[agent_type] = model
        if callbacks_out is not None:
            callbacks_out[agent_type] = callback

    return trained_models


def train_stage2(
    stage1_models: dict,
    preset_name: str | None = None,
    seed: int | None = None,
    callbacks_out: dict | None = None,
) -> PPO:
    """Train Policy Maker against frozen household policies."""
    models_dir = Path(config.PATHS["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stage 2: Training government (Policy Maker)...")
    env = make_env(preset_name)
    if seed is not None:
        env.reset(seed=seed)

    wrapper = PolicyMakerWrapper(env, frozen_models=stage1_models)
    model = _build_ppo(wrapper, "government")

    timesteps = config.TRAINING["stage2_timesteps"]
    callback = LoggingCallback("government")

    model.learn(total_timesteps=timesteps, callback=callback)

    save_path = config.PATHS["policymaker_policy"]
    model.save(save_path)
    logger.info(f"Stage 2: government saved to {save_path}.zip")

    if callbacks_out is not None:
        callbacks_out["government"] = callback

    return model


def load_models() -> dict:
    """Load all five trained policies from disk."""
    models = {}
    agent_types = ["displaced", "renter", "owner", "investor"]

    for agent_type in agent_types:
        path = config.PATHS[f"{agent_type}_policy"] + ".zip"
        if Path(path).exists():
            models[agent_type] = PPO.load(path)
            logger.info(f"Loaded {agent_type} from {path}")
        else:
            logger.warning(f"No saved model for {agent_type} at {path}")

    gov_path = config.PATHS["policymaker_policy"] + ".zip"
    if Path(gov_path).exists():
        models["government"] = PPO.load(gov_path)
        logger.info(f"Loaded government from {gov_path}")
    else:
        logger.warning(f"No saved model for government at {gov_path}")

    return models


def get_policy_callable(model: PPO) -> callable:
    """Return a function obs -> action for a trained SB3 model."""

    def _predict(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return _predict


def plot_training_curves(callbacks: dict, save_path: str = "results/training_curves.png"):
    """Generate and save training reward curves for all agent types."""
    import matplotlib.pyplot as plt

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    agent_types = list(callbacks.keys())
    colors = {
        "displaced": "#ff7f0e",
        "renter": "#1f77b4",
        "owner": "#2ca02c",
        "investor": "#d62728",
        "government": "#9467bd",
    }

    for idx, agent_type in enumerate(agent_types):
        cb = callbacks[agent_type]
        ax = axes[idx]

        if cb.timesteps_log:
            ax.plot(
                cb.timesteps_log, cb.rewards_log,
                color=colors.get(agent_type, "black"),
                linewidth=1.5,
            )
            ax.fill_between(
                cb.timesteps_log, cb.rewards_log,
                alpha=0.15, color=colors.get(agent_type, "black"),
            )
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean Episode Reward")
            ax.set_title(f"{agent_type.capitalize()} Agent")
            ax.grid(True, alpha=0.3)

            if len(cb.rewards_log) > 1:
                start_r = cb.rewards_log[0]
                end_r = cb.rewards_log[-1]
                ax.axhline(y=start_r, linestyle="--", alpha=0.4, color="gray")
                ax.annotate(
                    f"start: {start_r:.1f}",
                    xy=(cb.timesteps_log[0], start_r),
                    fontsize=8, color="gray",
                )
                ax.annotate(
                    f"final: {end_r:.1f}",
                    xy=(cb.timesteps_log[-1], end_r),
                    fontsize=8, color=colors.get(agent_type, "black"),
                    fontweight="bold",
                )
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{agent_type.capitalize()} Agent")

    if len(agent_types) < 6:
        axes[5].axis("off")

    fig.suptitle(
        f"Training Curves (PPO, {config.TRAINING['ppo_n_epochs']} epochs, "
        f"lr={config.TRAINING['ppo_learning_rate']})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {save_path}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train housing market RL agents")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick-test", action="store_true",
                        help="Override all timesteps to 500 for smoke testing")
    args = parser.parse_args()

    if args.quick_test:
        for key in config.TRAINING["stage1_timesteps"]:
            config.TRAINING["stage1_timesteps"][key] = 500
        config.TRAINING["stage2_timesteps"] = 500

    all_callbacks = {}

    if args.stage2_only:
        logger.info("Loading Stage 1 models for Stage 2...")
        stage1_models = load_models()
        stage1_models.pop("government", None)
        train_stage2(stage1_models, preset_name=args.preset, seed=args.seed,
                     callbacks_out=all_callbacks)
    elif args.stage1_only:
        train_stage1(preset_name=args.preset, seed=args.seed,
                     callbacks_out=all_callbacks)
    else:
        stage1_models = train_stage1(preset_name=args.preset, seed=args.seed,
                                     callbacks_out=all_callbacks)
        train_stage2(stage1_models, preset_name=args.preset, seed=args.seed,
                     callbacks_out=all_callbacks)

    if all_callbacks:
        plot_training_curves(all_callbacks)
