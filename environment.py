"""
Extended housing market environment.

Subclasses MultiAgentHousingEnv from the starter code, adding:
- Salary growth per step
- Seasonal demand spikes (non-mutating)
- Informal market mechanics
- Crisis injection (recession, supply shock, migration wave)
- Extended observations (+3 features per agent type)
- Reward computation via agents.py functions
- Extended history logging
"""

import numpy as np
from gymnasium import spaces

from problemA_starter_code import MultiAgentHousingEnv, House
import config
import agents as agents_module


class HousingMarketEnv(MultiAgentHousingEnv):

    def __init__(self, dormant_pool_size: int = 30, **kwargs):
        super().__init__(**kwargs)

        self.episode_count: int = 0
        self.seasonal_steps_remaining: int = 0
        self.informal_activity: float = 0.0
        self.prev_equity: dict = {}
        self.capital_gain: dict = {}
        self.price_history: list = []
        self.crisis_state: dict | None = None
        self._construction_halted: bool = False
        self.construction_config: dict = config.CONSTRUCTION
        self.extended_history: list = []
        self.dormant_pool_size: int = dormant_pool_size

        self._create_dormant_pool()

        for agent_id in self.possible_agents:
            if agent_id == "government":
                self.observation_spaces[agent_id] = spaces.Box(
                    low=0, high=np.inf, shape=(18,), dtype=np.float32
                )
            else:
                self.observation_spaces[agent_id] = spaces.Box(
                    low=0, high=np.inf, shape=(13,), dtype=np.float32
                )

    def _create_dormant_pool(self):
        existing_displaced = self.num_displaced
        for i in range(self.dormant_pool_size):
            agent_id = f"displaced_{existing_displaced + i}"
            if agent_id not in self.possible_agents:
                self.possible_agents.append(agent_id)
            if agent_id not in self.agents:
                self.agents.append(agent_id)
            self.action_spaces[agent_id] = spaces.Discrete(5)
            self.observation_spaces[agent_id] = spaces.Box(
                low=0, high=np.inf, shape=(13,), dtype=np.float32
            )
            self.agent_properties[agent_id] = {
                "type": "displaced",
                "wealth": 0.0,
                "income": 0.0,
                "housed": False,
                "house_id": None,
                "rent_payment": 0,
                "satisfaction": 0,
                "dormant": True,
            }

    def _initialize_agents(self):
        super()._initialize_agents()
        existing_displaced = self.num_displaced
        for i in range(self.dormant_pool_size):
            agent_id = f"displaced_{existing_displaced + i}"
            self.agent_properties[agent_id] = {
                "type": "displaced",
                "wealth": 0.0,
                "income": 0.0,
                "housed": False,
                "house_id": None,
                "rent_payment": 0,
                "satisfaction": 0,
                "dormant": True,
            }

    def _is_dormant(self, agent_id: str) -> bool:
        props = self.agent_properties.get(agent_id)
        if props is None:
            return True
        return props.get("dormant", False)

    def reset(self, seed=None, options=None):
        if self.crisis_state is not None:
            self.crisis_state["episodes_remaining"] -= 1
            if self.crisis_state["type"] == "recession":
                rate = self.crisis_state["params"]["recovery_rate_per_episode"]
                for agent_id in self.agents:
                    if agent_id == "government" or self._is_dormant(agent_id):
                        continue
                    self.agent_properties[agent_id]["income"] *= (1.0 + rate)
            if self.crisis_state["episodes_remaining"] <= 0:
                if self.crisis_state["type"] == "supply_shock":
                    self._construction_halted = False
                self.crisis_state = None

        observations, infos = super().reset(seed=seed, options=options)

        self._reinit_dormant_pool()

        self.prev_equity = {}
        for agent_id in self.agents:
            props = self.agent_properties.get(agent_id)
            if props and props.get("type") == "owner":
                self.prev_equity[agent_id] = props.get("equity", 0.0)

        self.capital_gain = {}
        for agent_id in self.agents:
            props = self.agent_properties.get(agent_id)
            if props and props.get("type") == "investor":
                self.capital_gain[agent_id] = 0.0

        self.seasonal_steps_remaining = 0
        self.informal_activity = 0.0
        self.price_history = []
        self.extended_history = []

        self.episode_count += 1
        if self.episode_count % config.SEASONAL_DEMAND["trigger_every_n_episodes"] == 0:
            self._tick_seasonal_spike()

        observations = self._get_extended_observations(observations)
        return observations, infos

    def _reinit_dormant_pool(self):
        existing_displaced = self.num_displaced
        for i in range(self.dormant_pool_size):
            agent_id = f"displaced_{existing_displaced + i}"
            self.agent_properties[agent_id] = {
                "type": "displaced",
                "wealth": 0.0,
                "income": 0.0,
                "housed": False,
                "house_id": None,
                "rent_payment": 0,
                "satisfaction": 0,
                "dormant": True,
            }
            if agent_id not in self.agents:
                self.agents.append(agent_id)

    def step(self, actions: dict):
        intercepted = {}
        filtered_actions = {}
        for agent_id, action in actions.items():
            if self._is_dormant(agent_id):
                continue
            if "investor" in agent_id and action == 4:
                intercepted[agent_id] = action
            else:
                filtered_actions[agent_id] = action

        for agent_id in intercepted:
            self._execute_informal_rental(agent_id)

        if self._construction_halted:
            saved_queue = self.construction_queue
            self.construction_queue = []
            observations, rewards, terminated, truncated, infos = super().step(
                filtered_actions
            )
            self.construction_queue = saved_queue
        else:
            observations, rewards, terminated, truncated, infos = super().step(
                filtered_actions
            )

        self._apply_salary_growth()
        self._decay_informal_activity()

        if self.seasonal_steps_remaining > 0:
            self.seasonal_steps_remaining -= 1

        for agent_id in list(self.prev_equity.keys()):
            props = self.agent_properties.get(agent_id)
            if props and props.get("type") == "owner":
                current_equity = props.get("equity", 0.0)
                self.prev_equity[agent_id] = current_equity

        for agent_id in list(self.capital_gain.keys()):
            props = self.agent_properties.get(agent_id)
            if props and props.get("type") == "investor":
                portfolio_value = sum(
                    h.price
                    for h in self.housing_stock
                    if h.owner == agent_id
                )
                prev_value = props.get("total_investment", 0.0)
                self.capital_gain[agent_id] = portfolio_value - prev_value

        avg_price = (
            np.mean([h.price for h in self.housing_stock])
            if self.housing_stock
            else 0.0
        )
        self.price_history.append(avg_price)

        env_state = self._build_env_state()

        if self.seasonal_steps_remaining > 0:
            env_state = self._apply_seasonal_to_env_state(env_state)

        rewards = self._compute_all_rewards(env_state)

        observations = self._get_extended_observations(observations)

        for agent_id in list(terminated.keys()):
            if self._is_dormant(agent_id):
                terminated[agent_id] = True
                truncated[agent_id] = False

        self._log_step(env_state, rewards)

        return observations, rewards, terminated, truncated, infos

    def _execute_government_action(self, action: int):
        if action == 3:
            cost = self.construction_config["cost_per_unit"]
            if self._construction_halted:
                return
            if self.agent_properties["government"]["budget"] >= cost:
                house = House(
                    self.house_price_avg * 1.2,
                    self.house_price_std,
                    self.house_rent_avg * self.construction_config["subsidized_rent_multiplier"],
                    self.house_rent_std,
                )
                duration = np.random.randint(
                    self.construction_config["delay_min_steps"],
                    self.construction_config["delay_max_steps"],
                )
                self.construction_queue.append(
                    {
                        "house": house,
                        "start_step": self.current_step,
                        "duration": duration,
                        "progress": 0,
                    }
                )
                self.agent_properties["government"]["budget"] -= cost
        else:
            super()._execute_government_action(action)

    def _build_env_state(self) -> dict:
        renter_incomes = []
        renter_rents = []
        displaced_count = 0
        total_household = 0

        for agent_id in self.agents:
            if self._is_dormant(agent_id) or agent_id == "government":
                continue
            props = self.agent_properties[agent_id]
            agent_type = props.get("type", "")
            total_household += 1

            if agent_type == "displaced" and not props.get("housed", False):
                displaced_count += 1
            if agent_type == "renter" and props.get("housed", False):
                income = props.get("income", 1.0)
                rent = props.get("rent_payment", 0.0)
                renter_incomes.append(income)
                renter_rents.append(rent)

        if renter_incomes:
            avg_rent_to_income = np.mean(
                [r / max(i, 1.0) for r, i in zip(renter_rents, renter_incomes)]
            )
        else:
            avg_rent_to_income = 0.0

        displacement_rate = (
            displaced_count / max(total_household, 1)
        )

        occupied = sum(
            1
            for h in self.housing_stock
            if h.status in ("owned", "rented")
        )
        occupancy_fraction = (
            occupied / max(len(self.housing_stock), 1)
        )

        avg_house_price = (
            np.mean([h.price for h in self.housing_stock])
            if self.housing_stock
            else 0.0
        )

        if len(self.price_history) >= 2:
            price_trend = (
                (self.price_history[-1] - self.price_history[-2])
                / max(self.price_history[-2], 1.0)
            )
        else:
            price_trend = 0.0

        for_rent_count = sum(
            1 for h in self.housing_stock if h.status == "for_rent"
        )
        rented_count = sum(
            1 for h in self.housing_stock if h.status == "rented"
        )
        total_rental = for_rent_count + rented_count
        if total_rental > 0:
            rental_demand_level = rented_count / total_rental
        else:
            rental_demand_level = 0.0

        return {
            "avg_rent_to_income": float(avg_rent_to_income),
            "displacement_rate": float(displacement_rate),
            "occupancy_fraction": float(occupancy_fraction),
            "avg_house_price": float(avg_house_price),
            "informal_activity": float(self.informal_activity),
            "price_trend": float(price_trend),
            "rental_demand_level": float(rental_demand_level),
        }

    def _compute_all_rewards(self, env_state: dict) -> dict:
        rewards = {}

        for agent_id in self.agents:
            if self._is_dormant(agent_id):
                rewards[agent_id] = 0.0
                continue

            props = self.agent_properties.get(agent_id, {})
            agent_type = props.get("type", "")

            if agent_id == "government":
                rewards[agent_id] = agents_module.compute_government_reward(
                    props, env_state
                )
            elif agent_type == "displaced":
                rewards[agent_id] = agents_module.compute_displaced_reward(
                    props, env_state
                )
            elif agent_type == "renter":
                rewards[agent_id] = agents_module.compute_renter_reward(
                    props, env_state
                )
            elif agent_type == "owner":
                prev_eq = self.prev_equity.get(agent_id, 0.0)
                rewards[agent_id] = agents_module.compute_owner_reward(
                    props, prev_eq, env_state
                )
            elif agent_type == "investor":
                cap_gain = self.capital_gain.get(agent_id, 0.0)
                rewards[agent_id] = agents_module.compute_investor_reward(
                    props, cap_gain, self.interest_rate, env_state
                )
            else:
                rewards[agent_id] = 0.0

        return rewards

    def _get_extended_observations(self, base_observations: dict) -> dict:
        env_state = self._build_env_state()

        if self.seasonal_steps_remaining > 0:
            env_state = self._apply_seasonal_to_env_state(env_state)

        extended = {}
        for agent_id, obs in base_observations.items():
            if self._is_dormant(agent_id):
                extended[agent_id] = np.zeros(13, dtype=np.float32)
                continue

            props = self.agent_properties.get(agent_id, {})
            agent_type = props.get("type", "")

            if agent_id == "government":
                agent_type = "government"

            ext_keys = agents_module.OBS_EXTENSIONS.get(agent_type)
            if ext_keys is None:
                extended[agent_id] = obs
                continue

            extra = np.array(
                [env_state.get(k, 0.0) for k in ext_keys], dtype=np.float32
            )
            extended[agent_id] = np.concatenate([obs, extra])

        return extended

    # ------------------------------------------------------------------
    # Extension helpers
    # ------------------------------------------------------------------

    def _apply_salary_growth(self) -> None:
        for agent_id in self.agents:
            if agent_id == "government" or self._is_dormant(agent_id):
                continue
            props = self.agent_properties[agent_id]
            agent_type = props.get("type", "")
            growth_rate = config.SALARY_GROWTH.get(agent_type, 0.0)
            props["income"] *= 1.0 + growth_rate / self.max_steps

    def _tick_seasonal_spike(self) -> None:
        self.seasonal_steps_remaining = config.SEASONAL_DEMAND[
            "spike_duration_steps"
        ]

    def _apply_seasonal_to_env_state(self, env_state: dict) -> dict:
        modified = dict(env_state)
        modified["avg_rent_to_income"] = (
            env_state["avg_rent_to_income"]
            * config.SEASONAL_DEMAND["rent_multiplier"]
        )
        modified["rental_demand_level"] = min(
            1.0,
            env_state["rental_demand_level"]
            * config.SEASONAL_DEMAND["competition_multiplier"],
        )
        return modified

    def _execute_informal_rental(self, agent_id: str) -> None:
        props = self.agent_properties[agent_id]
        owned_houses = props.get("owned_houses", [])
        if not owned_houses:
            return

        for_rent_houses = [
            h
            for h in self.housing_stock
            if h.id in owned_houses and h.status == "for_rent"
        ]
        if not for_rent_houses:
            return

        house = for_rent_houses[0]
        informal_rent = house.rent * config.INFORMAL_MARKET["above_ceiling_multiplier"]

        if np.random.random() < config.INFORMAL_MARKET["acceptance_base_probability"]:
            house.rent = informal_rent
            self.informal_activity = min(
                1.0,
                self.informal_activity
                + config.INFORMAL_MARKET["activity_increment"],
            )

    def _decay_informal_activity(self) -> None:
        self.informal_activity = max(
            0.0,
            self.informal_activity - config.INFORMAL_MARKET["activity_decay_rate"],
        )

    # ------------------------------------------------------------------
    # Crisis injection
    # ------------------------------------------------------------------

    def trigger_recession(self) -> None:
        params = dict(config.CRISIS["recession"])
        self.crisis_state = {
            "type": "recession",
            "episodes_remaining": params["duration_episodes"],
            "params": params,
        }
        drop = params["income_drop_fraction"]
        for agent_id in self.agents:
            if agent_id == "government" or self._is_dormant(agent_id):
                continue
            self.agent_properties[agent_id]["income"] *= 1.0 - drop

    def trigger_supply_shock(self) -> None:
        params = dict(config.CRISIS["supply_shock"])
        self.crisis_state = {
            "type": "supply_shock",
            "episodes_remaining": params["duration_episodes"],
            "params": params,
        }
        self._construction_halted = True

        units_to_remove = params["units_removed"]
        removed = 0

        vacant_houses = [
            h for h in self.housing_stock if h.status == "vacant"
        ]
        for house in vacant_houses:
            if removed >= units_to_remove:
                break
            self.housing_stock.remove(house)
            removed += 1

        if removed < units_to_remove:
            for_rent_houses = [
                h for h in self.housing_stock if h.status == "for_rent"
            ]
            for house in for_rent_houses:
                if removed >= units_to_remove:
                    break
                if house.tenant:
                    tenant_id = house.tenant
                    if tenant_id in self.agent_properties:
                        self.agent_properties[tenant_id]["housed"] = False
                        self.agent_properties[tenant_id]["house_id"] = None
                self.housing_stock.remove(house)
                removed += 1

    def trigger_migration_wave(self) -> None:
        params = config.CRISIS["migration_wave"]
        count = params["new_displaced_count"]
        activated = 0

        for agent_id in self.agents:
            if activated >= count:
                break
            props = self.agent_properties.get(agent_id)
            if props and props.get("dormant", False):
                props["dormant"] = False
                props["income"] = self.income_avg * params["income_multiplier"]
                props["wealth"] = self.wealth_avg * params["wealth_multiplier"]
                props["housed"] = False
                props["house_id"] = None
                props["satisfaction"] = 0
                activated += 1

    # ------------------------------------------------------------------
    # History logging
    # ------------------------------------------------------------------

    def _log_step(self, env_state: dict, rewards: dict) -> None:
        rewards_by_type = {
            "displaced": [],
            "renter": [],
            "owner": [],
            "investor": [],
            "government": [],
        }
        agent_count_by_type = {
            "displaced": 0,
            "renter": 0,
            "owner": 0,
            "investor": 0,
            "government": 0,
        }

        for agent_id, reward in rewards.items():
            if self._is_dormant(agent_id):
                continue
            props = self.agent_properties.get(agent_id, {})
            if agent_id == "government":
                atype = "government"
            else:
                atype = props.get("type", "")
            if atype in rewards_by_type:
                rewards_by_type[atype].append(reward)
                agent_count_by_type[atype] += 1

        mean_rewards = {
            k: float(np.mean(v)) if v else 0.0
            for k, v in rewards_by_type.items()
        }

        self.extended_history.append(
            {
                "step": self.current_step,
                "avg_house_price": env_state["avg_house_price"],
                "avg_rent_to_income": env_state["avg_rent_to_income"],
                "displacement_rate": env_state["displacement_rate"],
                "occupancy_fraction": env_state["occupancy_fraction"],
                "informal_activity": env_state["informal_activity"],
                "price_trend": env_state["price_trend"],
                "rental_demand_level": env_state["rental_demand_level"],
                "interest_rate": self.interest_rate,
                "construction_queue_size": len(self.construction_queue),
                "seasonal_active": self.seasonal_steps_remaining > 0,
                "crisis_type": (
                    self.crisis_state["type"] if self.crisis_state else None
                ),
                "rewards_by_type": mean_rewards,
                "agent_count_by_type": agent_count_by_type,
            }
        )
