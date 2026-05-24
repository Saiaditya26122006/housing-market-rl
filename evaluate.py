"""
Experiment runner for the multi-agent housing market simulation.

Runs three comparison scenarios (random, heuristic, trained) each in
baseline and stressed conditions, saves results to JSON.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from utils import (
    make_env,
    load_all_models,
    run_episode,
    compute_summary_metrics,
    serialize_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CRISIS_SCHEDULE = [
    (20, "recession"),
    (50, "supply_shock"),
    (75, "migration_wave"),
]


def run_scenario(
    mode: str,
    n_episodes: int,
    preset: str | None = None,
    crisis_schedule: list | None = None,
    models: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Run N episodes in the given mode with optional crisis schedule."""
    models = models or {}
    episodes = []

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx if seed is not None else None
        env = make_env(preset)
        if ep_seed is not None:
            env.reset(seed=ep_seed)

        result = run_episode(env, models, mode, crisis_schedule=crisis_schedule)
        result["episode_id"] = ep_idx
        episodes.append(result)

        logger.info(
            f"  [{mode}] episode {ep_idx + 1}/{n_episodes} "
            f"displacement={result['summary']['mean_displacement_rate']:.3f} "
            f"occupancy={result['summary']['final_occupancy']:.3f}"
        )

    aggregate = _compute_aggregate(episodes)

    return {
        "episodes": episodes,
        "aggregate": aggregate,
    }


def _compute_aggregate(episodes: list) -> dict:
    """Compute mean and std of summary metrics across episodes."""
    if not episodes:
        return {}

    keys = [
        "mean_displacement_rate",
        "mean_rent_to_income",
        "final_occupancy",
        "price_volatility",
    ]

    aggregate = {}
    for key in keys:
        values = [ep["summary"][key] for ep in episodes]
        aggregate[f"mean_{key}"] = float(np.mean(values))
        aggregate[f"std_{key}"] = float(np.std(values))

    reward_types = set()
    for ep in episodes:
        reward_types.update(ep["summary"].get("mean_rewards", {}).keys())

    aggregate["mean_rewards"] = {}
    for rtype in sorted(reward_types):
        values = [
            ep["summary"]["mean_rewards"].get(rtype, 0.0) for ep in episodes
        ]
        aggregate["mean_rewards"][rtype] = float(np.mean(values))

    return aggregate


def run_all_scenarios(
    n_episodes: int = 10,
    preset: str | None = None,
    seed: int | None = None,
) -> dict:
    """Run all 3 modes x 2 conditions (baseline + stressed)."""
    models = load_all_models()

    scenarios = {}
    modes = ["random", "heuristic", "trained"]

    for mode in modes:
        logger.info(f"Running {mode} baseline...")
        scenarios[f"{mode}_baseline"] = run_scenario(
            mode=mode,
            n_episodes=n_episodes,
            preset=preset,
            crisis_schedule=None,
            models=models,
            seed=seed,
        )

        logger.info(f"Running {mode} stressed...")
        scenarios[f"{mode}_stressed"] = run_scenario(
            mode=mode,
            n_episodes=n_episodes,
            preset=preset,
            crisis_schedule=CRISIS_SCHEDULE,
            models=models,
            seed=seed,
        )

    return {
        "metadata": {
            "preset": preset or "default",
            "episodes_per_scenario": n_episodes,
            "crisis_schedule": CRISIS_SCHEDULE,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "scenarios": scenarios,
    }


def print_comparison_table(results: dict) -> None:
    """Log a formatted comparison table of key metrics."""
    scenarios = results["scenarios"]

    header = f"{'Scenario':<25} {'Displ.Rate':>10} {'Rent/Inc':>10} {'Occupancy':>10} {'PriceVol':>10}"
    logger.info("")
    logger.info("=" * 70)
    logger.info(header)
    logger.info("-" * 70)

    for name, data in scenarios.items():
        agg = data["aggregate"]
        logger.info(
            f"{name:<25} "
            f"{agg.get('mean_mean_displacement_rate', 0):.4f}     "
            f"{agg.get('mean_mean_rent_to_income', 0):.4f}     "
            f"{agg.get('mean_final_occupancy', 0):.4f}     "
            f"{agg.get('mean_price_volatility', 0):.4f}"
        )

    logger.info("=" * 70)
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation scenarios")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["random", "heuristic", "trained", "all"],
    )
    parser.add_argument("--no-crisis", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(config.PATHS["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario == "all":
        results = run_all_scenarios(
            n_episodes=args.episodes,
            preset=args.preset,
            seed=args.seed,
        )
        print_comparison_table(results)

        output_path = str(results_dir / "evaluation_results.json")
        serialize_results(results, output_path)
    else:
        models = load_all_models()
        crisis = None if args.no_crisis else CRISIS_SCHEDULE

        logger.info(f"Running {args.scenario} baseline...")
        baseline = run_scenario(
            mode=args.scenario,
            n_episodes=args.episodes,
            preset=args.preset,
            crisis_schedule=None,
            models=models,
            seed=args.seed,
        )

        results = {
            "metadata": {
                "preset": args.preset or "default",
                "episodes_per_scenario": args.episodes,
                "crisis_schedule": crisis,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "scenarios": {f"{args.scenario}_baseline": baseline},
        }

        if not args.no_crisis:
            logger.info(f"Running {args.scenario} stressed...")
            stressed = run_scenario(
                mode=args.scenario,
                n_episodes=args.episodes,
                preset=args.preset,
                crisis_schedule=crisis,
                models=models,
                seed=args.seed,
            )
            results["scenarios"][f"{args.scenario}_stressed"] = stressed

        path_key = f"scenario_{args.scenario}"
        output_path = config.PATHS.get(
            path_key, str(results_dir / f"scenario_{args.scenario}.json")
        )
        if not output_path.endswith(".json"):
            output_path += ".json"
        serialize_results(results, output_path)


if __name__ == "__main__":
    main()
