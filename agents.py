"""
Agent observation extensions, reward functions, and heuristic policies.

env_state contract — environment.py must pass a dict with these keys:
    avg_rent_to_income: float — average rent / average income across renters
    displacement_rate: float — fraction of household agents currently displaced
    occupancy_fraction: float — fraction of houses currently occupied
    avg_house_price: float — mean house price in the market
    informal_activity: float — level of informal market activity [0, 1]
    price_trend: float — rolling price change direction (positive = rising)
    rental_demand_level: float — normalized rental demand pressure [0, 1]

Cross-step state requirement:
    environment.py must maintain two dicts updated each step:
        prev_equity: dict[int, float] — previous step equity per owner agent index
        capital_gain: dict[int, float] — realized/unrealized capital gain per investor index
"""

import numpy as np

import config


# ---------------------------------------------------------------------------
# Section 1: OBS_EXTENSIONS
# Three additional features appended to each agent type's observation vector.
# Household agents: base 10 -> 13. Government: base 15 -> 18.
# ---------------------------------------------------------------------------

OBS_EXTENSIONS: dict = {
    "displaced": ["price_trend", "rental_demand_level", "informal_activity"],
    "renter": ["avg_rent_to_income", "price_trend", "informal_activity"],
    "owner": ["price_trend", "occupancy_fraction", "avg_house_price"],
    "investor": ["price_trend", "rental_demand_level", "occupancy_fraction"],
    "government": ["informal_activity", "price_trend", "rental_demand_level"],
}


# ---------------------------------------------------------------------------
# Section 2: OBS_SLICE
# Index range for each agent type's full observation vector.
# ---------------------------------------------------------------------------

OBS_SLICE: dict = {
    "displaced": slice(0, 13),
    "renter": slice(0, 13),
    "owner": slice(0, 13),
    "investor": slice(0, 13),
    "government": slice(0, 18),
}


# ---------------------------------------------------------------------------
# Section 3: Reward functions
# Each reads scaling constants exclusively from config.REWARD_WEIGHTS.
# ---------------------------------------------------------------------------


def compute_displaced_reward(agent_props: dict, env_state: dict) -> float:
    """Reward for a displaced agent."""
    w = config.REWARD_WEIGHTS["displaced"]
    reward = 0.0

    if agent_props.get("housed_via_buy"):
        reward += w["housed_buy_reward"]
    elif agent_props.get("housed_via_rent"):
        reward += w["housed_rent_reward"]
    else:
        steps_displaced = agent_props.get("steps_displaced", 1)
        reward += w["step_displaced_penalty"] * (
            1.0 + w["duration_multiplier"] * steps_displaced
        )

    return reward


def compute_renter_reward(agent_props: dict, env_state: dict) -> float:
    """Reward for a renter agent."""
    w = config.REWARD_WEIGHTS["renter"]
    reward = 0.0

    if agent_props.get("transitioned_to_owner"):
        reward += w["ownership_transition_bonus"]
    elif agent_props.get("forced_move"):
        reward += w["forced_move_penalty"]
    elif agent_props.get("stable"):
        reward += w["stable_housing_reward"]

    rent_to_income = agent_props.get("rent_to_income", 0.0)
    if rent_to_income > w["rent_burden_threshold"]:
        excess = rent_to_income - w["rent_burden_threshold"]
        reward -= w["rent_burden_scale"] * excess

    return reward


def compute_owner_reward(
    agent_props: dict, prev_equity: float, env_state: dict
) -> float:
    """Reward for an owner agent."""
    w = config.REWARD_WEIGHTS["owner"]
    reward = 0.0

    current_equity = agent_props.get("equity", 0.0)
    equity_growth = current_equity - prev_equity
    reward += w["equity_growth_scale"] * equity_growth

    rental_income = agent_props.get("rental_income", 0.0)
    if rental_income > 0:
        reward += w["rental_income_bonus"] * rental_income

    return reward


def compute_investor_reward(
    agent_props: dict,
    capital_gain: float,
    interest_rate: float,
    env_state: dict,
) -> float:
    """Reward for an investor agent."""
    w = config.REWARD_WEIGHTS["investor"]
    reward = 0.0

    rental_income = agent_props.get("rental_income", 0.0)
    reward += w["rental_income_scale"] * rental_income

    mortgage_balance = agent_props.get("mortgage_balance", 0.0)
    reward += w["mortgage_cost_scale"] * (mortgage_balance * interest_rate)

    maintenance = agent_props.get("maintenance_cost", 0.0)
    reward += w["maintenance_cost_scale"] * maintenance

    reward += w["capital_gains_scale"] * capital_gain

    vacancy_count = agent_props.get("vacancy_count", 0)
    reward += w["vacancy_penalty"] * vacancy_count

    return reward


def compute_government_reward(agent_props: dict, env_state: dict) -> float:
    """Reward for the government agent."""
    w = config.REWARD_WEIGHTS["government"]

    displacement_score = (1.0 - env_state["displacement_rate"]) * w["displacement_scale"]

    rent_ratio = env_state["avg_rent_to_income"]
    if rent_ratio <= w["rent_income_threshold"]:
        affordability_score = w["rent_affordability_scale"]
    else:
        excess = rent_ratio - w["rent_income_threshold"]
        affordability_score = w["rent_affordability_scale"] * (1.0 - excess)

    occupancy = env_state["occupancy_fraction"]
    if occupancy >= w["supply_utilization_threshold"]:
        supply_score = w["supply_scale"]
    else:
        supply_score = w["supply_scale"] * (
            occupancy / w["supply_utilization_threshold"]
        )

    reward = (
        w["displacement_weight"] * displacement_score
        + w["rent_affordability_weight"] * affordability_score
        + w["supply_weight"] * supply_score
    )

    return reward


# ---------------------------------------------------------------------------
# Section 4: Heuristic policy classes
# Baselines for evaluation. Each has a predict(obs) -> int method.
# Action indices correspond to environment.py's existing Discrete spaces.
# ---------------------------------------------------------------------------


class DisplacedHeuristic:
    """Heuristic: attempt to buy if affordable, else rent, else wait."""

    def predict(self, obs: np.ndarray) -> int:
        income = obs[0]
        wealth = obs[1]
        avg_price = obs[3] if len(obs) > 3 else 0.0
        if wealth > avg_price * 0.2:
            return 0  # search_buy
        if income > 0:
            return 1  # search_rent
        return 2  # wait


class RenterHeuristic:
    """Heuristic: attempt to buy if can afford, stay if rent is manageable, else move."""

    def predict(self, obs: np.ndarray) -> int:
        income = obs[0]
        wealth = obs[1]
        rent = obs[2] if len(obs) > 2 else 0.0
        avg_price = obs[3] if len(obs) > 3 else 0.0
        if wealth > avg_price * 0.2:
            return 2  # attempt_buy
        rent_ratio = rent / income if income > 0 else 1.0
        if rent_ratio > 0.4:
            return 1  # move
        return 0  # stay


class OwnerHeuristic:
    """Heuristic: rent out if price trend is flat, sell if very high, else hold."""

    def predict(self, obs: np.ndarray) -> int:
        price_trend = obs[10] if len(obs) > 10 else 0.0
        occupancy = obs[11] if len(obs) > 11 else 0.5
        if price_trend > 0.1:
            return 1  # sell
        if occupancy < 0.8:
            return 2  # rent_out
        return 0  # hold


class InvestorHeuristic:
    """Heuristic: buy when prices dropping, sell when rising, else hold."""

    def predict(self, obs: np.ndarray) -> int:
        price_trend = obs[10] if len(obs) > 10 else 0.0
        rental_demand = obs[11] if len(obs) > 11 else 0.5
        if price_trend < -0.05:
            return 0  # buy
        if price_trend > 0.1:
            return 1  # sell
        if rental_demand > 0.7:
            return 3  # set_rent (raise)
        return 2  # hold


class GovernmentHeuristic:
    """Heuristic: react to worst metric — displacement, affordability, or supply."""

    def predict(self, obs: np.ndarray) -> int:
        displacement_rate = obs[5] if len(obs) > 5 else 0.0
        rent_to_income = obs[6] if len(obs) > 6 else 0.0
        occupancy = obs[7] if len(obs) > 7 else 0.5

        if displacement_rate > 0.3:
            return 1  # subsidize_construction
        if rent_to_income > 0.4:
            return 0  # adjust_rent_ceiling
        if occupancy < 0.8:
            return 3  # relax_zoning
        return 2  # set_tax_rate
