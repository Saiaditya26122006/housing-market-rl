"""
Central configuration for the multi-agent housing market simulation.
All tunable parameters live here — no hardcoded values elsewhere.
"""


ENV_DEFAULTS: dict = {
    "num_displaced": 10,
    "num_renters": 20,
    "num_owners": 30,
    "num_investors": 5,
    "num_houses": 60,
    "income_avg": 45000,
    "income_std": 15000,
    "wealth_avg": 100000,
    "wealth_std": 50000,
    "house_price_avg": 300000,
    "house_price_std": 75000,
    "house_rent_avg": 1200,
    "house_rent_std": 300,
    "max_steps": 100,
    "interest_rate_start": 0.05,
}


SALARY_GROWTH: dict = {
    "displaced": 0.005,
    "renter": 0.008,
    "owner": 0.015,
    "investor": 0.020,
}


CONSTRUCTION: dict = {
    "delay_min_steps": 5,
    "delay_max_steps": 15,
    "cost_per_unit": 200000,
    "subsidized_rent_multiplier": 0.7,
}


SEASONAL_DEMAND: dict = {
    "trigger_every_n_episodes": 4,
    "spike_duration_steps": 10,
    "rent_multiplier": 1.2,
    "competition_multiplier": 1.3,
}


INFORMAL_MARKET: dict = {
    "above_ceiling_multiplier": 1.3,
    "acceptance_base_probability": 0.4,
    "activity_decay_rate": 0.05,
    "activity_increment": 0.1,
}


CRISIS: dict = {
    "recession": {
        "income_drop_fraction": 0.15,
        "duration_episodes": 5,
        "recovery_rate_per_episode": 0.03,
    },
    "supply_shock": {
        "units_removed": 10,
        "halt_construction": True,
        "duration_episodes": 3,
    },
    "migration_wave": {
        "new_displaced_count": 20,
        "income_multiplier": 0.7,
        "wealth_multiplier": 0.5,
    },
}


REWARD_WEIGHTS: dict = {
    "displaced": {
        "housed_buy_reward": 10.0,
        "housed_rent_reward": 5.0,
        "step_displaced_penalty": -1.0,
        "duration_multiplier": 0.1,
    },
    "renter": {
        "stable_housing_reward": 3.0,
        "ownership_transition_bonus": 8.0,
        "forced_move_penalty": -5.0,
        "rent_burden_threshold": 0.3,
        "rent_burden_scale": 2.0,
    },
    "owner": {
        "equity_growth_scale": 1.0,
        "rental_income_bonus": 2.0,
    },
    "investor": {
        "rental_income_scale": 1.5,
        "mortgage_cost_scale": -1.0,
        "maintenance_cost_scale": -0.5,
        "capital_gains_scale": 2.0,
        "vacancy_penalty": -3.0,
    },
    "government": {
        "displacement_weight": 0.5,
        "rent_affordability_weight": 0.3,
        "supply_weight": 0.2,
        "rent_income_threshold": 0.35,
        "supply_utilization_threshold": 0.85,
        "displacement_scale": 10.0,
        "rent_affordability_scale": 5.0,
        "supply_scale": 3.0,
    },
}


TRAINING: dict = {
    "stage1_timesteps": {
        "displaced": 50000,
        "renter": 50000,
        "owner": 50000,
        "investor": 50000,
        "government": 100000,
    },
    "stage2_timesteps": 200000,
    "ppo_learning_rate": 3e-4,
    "ppo_n_steps": 2048,
    "ppo_batch_size": 64,
    "ppo_n_epochs": 10,
    "net_arch": [64, 64],
    "device": "cpu",
}


PATHS: dict = {
    "models_dir": "models",
    "results_dir": "results",
    "displaced_policy": "models/displaced_policy",
    "renter_policy": "models/renter_policy",
    "owner_policy": "models/owner_policy",
    "investor_policy": "models/investor_policy",
    "policymaker_policy": "models/policymaker_policy",
    "scenario_random": "results/scenario_random",
    "scenario_heuristic": "results/scenario_heuristic",
    "scenario_trained": "results/scenario_trained",
}


CITY_PRESETS: dict = {
    "default": {
        "num_displaced": 10,
        "num_renters": 20,
        "num_owners": 30,
        "num_investors": 5,
        "num_houses": 60,
        "income_avg": 45000,
        "income_std": 15000,
        "wealth_avg": 100000,
        "wealth_std": 50000,
        "house_price_avg": 300000,
        "house_price_std": 75000,
        "house_rent_avg": 1200,
        "house_rent_std": 300,
        "max_steps": 100,
        "interest_rate_start": 0.05,
    },
    "madrid": {
        "income_avg": 30000,  # INE Spain Encuesta de Estructura Salarial 2022
        "income_std": 12000,
        "wealth_avg": 150000,  # Banco de España Encuesta Financiera de las Familias 2020
        "wealth_std": 70000,
        "house_price_avg": 350000,  # INE Spain Indice de Precios de Vivienda Q4 2023
        "house_price_std": 90000,
        "house_rent_avg": 1400,  # Eurostat Housing Cost Statistics 2023
        "house_rent_std": 350,
        "interest_rate_start": 0.045,  # ECB Statistical Data Warehouse Q4 2023
        "num_owners": 35,  # Eurostat Housing Statistics 2023 Spain homeownership rate 75 percent
        "num_renters": 22,
        "num_displaced": 15,  # INE Spain Encuesta de Condiciones de Vida 2023
        "num_investors": 8,
        "num_houses": 70,  # Madrid vacancy rate 1 to 2 percent Eurostat
        "max_steps": 100,
    },
}


def get_preset(name: str) -> dict:
    """Return a copy of the named city preset."""
    if name not in CITY_PRESETS:
        valid = ", ".join(sorted(CITY_PRESETS.keys()))
        raise ValueError(
            f"Unknown preset '{name}'. Valid presets: {valid}"
        )
    return {k: v for k, v in CITY_PRESETS[name].items()}
