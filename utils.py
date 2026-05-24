"""
Shared helpers for evaluate.py, dashboard.py, and train.py.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import agents as agents_module
import config
from environment import HousingMarketEnv

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


def make_env(preset_name: str | None = None) -> HousingMarketEnv:
    """Create a HousingMarketEnv from a preset or defaults."""
    if preset_name:
        params = config.get_preset(preset_name)
    else:
        params = dict(config.ENV_DEFAULTS)
    interest_rate = params.pop("interest_rate_start", 0.05)
    env = HousingMarketEnv(**params)
    env.interest_rate = interest_rate
    return env


def load_all_models() -> dict:
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


def get_action(
    agent_id: str,
    obs: np.ndarray,
    models: dict,
    mode: str,
    action_space=None,
) -> int:
    """Return an action based on mode: 'random', 'heuristic', or 'trained'."""
    agent_type = _get_agent_type(agent_id)

    if mode == "random":
        if action_space is not None:
            return int(action_space.sample())
        return 0

    if mode == "heuristic":
        return HEURISTICS[agent_type].predict(obs)

    if mode == "trained":
        model = models.get(agent_type)
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
        return HEURISTICS[agent_type].predict(obs)

    return 0


def run_episode(
    env: HousingMarketEnv,
    models: dict,
    mode: str,
    crisis_schedule: list | None = None,
) -> dict:
    """Run one full episode, injecting crises at scheduled steps."""
    observations, _ = env.reset()
    crisis_schedule = crisis_schedule or []
    crisis_events_fired = []

    done = False
    while not done:
        for scheduled_step, crisis_type in crisis_schedule:
            if env.current_step == scheduled_step:
                if crisis_type == "recession":
                    env.trigger_recession()
                elif crisis_type == "supply_shock":
                    env.trigger_supply_shock()
                elif crisis_type == "migration_wave":
                    env.trigger_migration_wave()
                crisis_events_fired.append(
                    {"step": scheduled_step, "type": crisis_type}
                )

        actions = {}
        for agent_id in env.agents:
            obs = observations.get(
                agent_id,
                np.zeros(
                    env.observation_spaces[agent_id].shape, dtype=np.float32
                ),
            )
            actions[agent_id] = get_action(
                agent_id, obs, models, mode, env.action_spaces[agent_id]
            )

        observations, rewards, terminated, truncated, infos = env.step(actions)
        done = all(terminated.values())

    summary = compute_summary_metrics(env.extended_history)
    summary["crisis_events"] = crisis_events_fired

    return {
        "steps": list(env.extended_history),
        "summary": summary,
    }


def compute_summary_metrics(episode_history: list) -> dict:
    """Compute aggregate metrics from an episode's extended_history."""
    if not episode_history:
        return {
            "mean_displacement_rate": 0.0,
            "mean_rent_to_income": 0.0,
            "final_occupancy": 0.0,
            "price_volatility": 0.0,
            "total_rewards": {},
            "mean_rewards": {},
        }

    displacement_rates = [s["displacement_rate"] for s in episode_history]
    rent_to_incomes = [s["avg_rent_to_income"] for s in episode_history]
    prices = [s["avg_house_price"] for s in episode_history]

    total_rewards = {}
    counts = {}
    for step_data in episode_history:
        for atype, mean_r in step_data.get("rewards_by_type", {}).items():
            agent_count = step_data.get("agent_count_by_type", {}).get(atype, 1)
            total_rewards[atype] = total_rewards.get(atype, 0.0) + mean_r * agent_count
            counts[atype] = counts.get(atype, 0) + agent_count

    mean_rewards = {
        k: total_rewards[k] / max(counts[k], 1) for k in total_rewards
    }

    price_array = np.array(prices)
    if len(price_array) > 1:
        returns = np.diff(price_array) / np.maximum(price_array[:-1], 1.0)
        price_volatility = float(np.std(returns))
    else:
        price_volatility = 0.0

    return {
        "mean_displacement_rate": float(np.mean(displacement_rates)),
        "mean_rent_to_income": float(np.mean(rent_to_incomes)),
        "final_occupancy": float(episode_history[-1]["occupancy_fraction"]),
        "price_volatility": price_volatility,
        "total_rewards": {k: float(v) for k, v in total_rewards.items()},
        "mean_rewards": {k: float(v) for k, v in mean_rewards.items()},
    }


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def serialize_results(results: dict, path: str) -> None:
    """Write results dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, cls=_NumpyEncoder, indent=2)
    logger.info(f"Results saved to {path}")


def load_results(path: str) -> dict:
    """Load results JSON from disk."""
    with open(path, "r") as f:
        return json.load(f)
