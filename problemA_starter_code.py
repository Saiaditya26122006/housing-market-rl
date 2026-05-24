import numpy as np
import matplotlib.pyplot as plt
from pettingzoo import ParallelEnv
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional


class House:
    """A class representing a house asset."""
    next_id = 1

    def __init__(self, h_price: float, h_price_std: float, h_rent: float, h_rent_std: float):
        self.id = House.next_id
        House.next_id += 1
        self.price = max(1000, np.random.normal(h_price, h_price_std))
        self.quality = np.random.uniform(1, 10)
        self.rent = max(100, np.random.normal(h_rent, h_rent_std))
        self.owner = None
        self.status = "vacant"  # "vacant", "owned", "for_rent", "rented"
        self.tenant = None
        self.rent_history = []


class MultiAgentHousingEnv(ParallelEnv):
    """
    Multi-agent housing market environment with 5 agent types:
    - government: Balance market stability and affordability
    - displaced: Find housing (buy or rent)
    - renters: Optimize housing costs and quality
    - owners: Maintain property value and potentially invest
    - investors: Maximize return on investment
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "housing_market_v1"}

    def __init__(self,
                 num_displaced: int = 20,
                 num_renters: int = 30,
                 num_owners: int = 25,
                 num_investors: int = 10,
                 num_houses: int = 80,
                 income_avg: float = 50000,
                 income_std: float = 15000,
                 wealth_avg: float = 100000,
                 wealth_std: float = 50000,
                 house_price_avg: float = 200000,
                 house_price_std: float = 50000,
                 house_rent_avg: float = 1500,
                 house_rent_std: float = 500,
                 max_steps: int = 100,
                 render_mode: Optional[str] = None):

        super().__init__()

        self.num_displaced = num_displaced
        self.num_renters = num_renters
        self.num_owners = num_owners
        self.num_investors = num_investors
        self.num_houses = num_houses
        self.income_avg = income_avg
        self.income_std = income_std
        self.wealth_avg = wealth_avg
        self.wealth_std = wealth_std
        self.house_price_avg = house_price_avg
        self.house_price_std = house_price_std
        self.house_rent_avg = house_rent_avg
        self.house_rent_std = house_rent_std
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.construction_queue = []

        # Create agent IDs
        self.possible_agents = (
                ["government"] +
                [f"displaced_{i}" for i in range(num_displaced)] +
                [f"renter_{i}" for i in range(num_renters)] +
                [f"owner_{i}" for i in range(num_owners)] +
                [f"investor_{i}" for i in range(num_investors)]
        )
        self.agents = self.possible_agents[:]

        # Agent properties
        self.agent_properties = {}
        self.housing_stock: List[House] = []
        self.current_step = 0
        self.interest_rate = 0.05
        self.rent_control_active = False

        # History tracking
        self.history = {
            'steps': [],
            'displaced_housed': [],
            'displaced_total': [],
            'renter_satisfaction': [],
            'owner_equity': [],
            'investor_returns': [],
            'market_stability': []
        }

        self._setup_action_spaces()
        self._setup_observation_spaces()

    def _setup_action_spaces(self):
        """Define action spaces for each agent type"""
        self.action_spaces = {}

        # Government actions: [do_nothing, lower_interest_rate, raise_interest_rate,
        #                     build_houses, rent_control_on, rent_control_off]
        self.action_spaces["government"] = spaces.Discrete(6)

        # Displaced actions: [do_nothing, try_buy_cheapest, try_rent_cheapest,
        #                    try_buy_best_value, try_rent_best_value]
        for agent in self.possible_agents:
            if "displaced" in agent:
                self.action_spaces[agent] = spaces.Discrete(5)

        # Renter actions: [stay, move_to_cheaper, move_to_better, try_to_buy]
        for agent in self.possible_agents:
            if "renter" in agent:
                self.action_spaces[agent] = spaces.Discrete(4)

        # Owner actions: [do_nothing, sell_house, rent_out_house, buy_another]
        for agent in self.possible_agents:
            if "owner" in agent:
                self.action_spaces[agent] = spaces.Discrete(4)

        # Investor actions: [do_nothing, buy_house, sell_house, adjust_rent_up, adjust_rent_down]
        for agent in self.possible_agents:
            if "investor" in agent:
                self.action_spaces[agent] = spaces.Discrete(5)

    def _setup_observation_spaces(self):
        """Define observation spaces for each agent type"""
        self.observation_spaces = {}

        # Government: Full market view (15 features)
        self.observation_spaces["government"] = spaces.Box(
            low=0, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Individual agents: Personal + limited market info (10 features)
        for agent in self.possible_agents:
            if agent != "government":
                self.observation_spaces[agent] = spaces.Box(
                    low=0, high=np.inf, shape=(10,), dtype=np.float32
                )

    def _initialize_agents(self):
        """Initialize agent properties"""
        self.agent_properties = {}

        # Government
        self.agent_properties["government"] = {
            "type": "government",
            "budget": 1000000,  # Government budget for building houses
        }

        # Displaced agents
        for i in range(self.num_displaced):
            agent_id = f"displaced_{i}"
            self.agent_properties[agent_id] = {
                "type": "displaced",
                "wealth": max(0, np.random.normal(self.wealth_avg * 0.3, self.wealth_std * 0.5)),
                "income": abs(np.random.normal(self.income_avg * 0.6, self.income_std * 0.5)),
                "housed": False,
                "house_id": None,
                "rent_payment": 0,
                "satisfaction": 0
            }

        # Renter agents
        for i in range(self.num_renters):
            agent_id = f"renter_{i}"
            self.agent_properties[agent_id] = {
                "type": "renter",
                "wealth": max(0, np.random.normal(self.wealth_avg * 0.7, self.wealth_std * 0.6)),
                "income": abs(np.random.normal(self.income_avg * 0.8, self.income_std * 0.6)),
                "housed": True,
                "house_id": None,  # Will be assigned during setup
                "rent_payment": 0,
                "satisfaction": 5  # Start with medium satisfaction
            }

        # Owner agents
        for i in range(self.num_owners):
            agent_id = f"owner_{i}"
            self.agent_properties[agent_id] = {
                "type": "owner",
                "wealth": max(0, np.random.normal(self.wealth_avg, self.wealth_std)),
                "income": abs(np.random.normal(self.income_avg, self.income_std)),
                "housed": True,
                "owned_houses": [],  # Will own at least one house
                "house_id": None,  # Primary residence
                "equity": 0
            }

        # Investor agents
        for i in range(self.num_investors):
            agent_id = f"investor_{i}"
            self.agent_properties[agent_id] = {
                "type": "investor",
                "wealth": max(0, np.random.normal(self.wealth_avg * 2, self.wealth_std)),
                "income": abs(np.random.normal(self.income_avg * 1.5, self.income_std)),
                "owned_houses": [],
                "rental_income": 0,
                "total_investment": 0
            }

    def _create_housing_stock(self):
        """Create initial housing stock and assign some to existing agents"""
        self.housing_stock = []
        House.next_id = 1

        for _ in range(self.num_houses):
            house = House(
                self.house_price_avg, self.house_price_std,
                self.house_rent_avg, self.house_rent_std
            )
            self.housing_stock.append(house)

        # Assign houses to owners (they start with houses)
        owner_agents = [a for a in self.agents if "owner" in a]
        available_houses = [h for h in self.housing_stock if h.status == "vacant"]

        for i, agent_id in enumerate(owner_agents):
            if i < len(available_houses):
                house = available_houses[i]
                house.status = "owned"
                house.owner = agent_id
                self.agent_properties[agent_id]["owned_houses"] = [house.id]
                self.agent_properties[agent_id]["house_id"] = house.id
                self.agent_properties[agent_id]["equity"] = house.price

        # Some investors start with rental properties
        investor_agents = [a for a in self.agents if "investor" in a]
        remaining_houses = [h for h in self.housing_stock if h.status == "vacant"]

        for i, agent_id in enumerate(investor_agents):
            if i < len(remaining_houses) // 2:  # Half of investors start with property
                house = remaining_houses[i]
                house.status = "for_rent"
                house.owner = agent_id
                self.agent_properties[agent_id]["owned_houses"] = [house.id]
                self.agent_properties[agent_id]["total_investment"] = house.price

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents"""
        observations = {}

        # Market statistics
        avg_house_price = np.mean([h.price for h in self.housing_stock])
        avg_rent = np.mean([h.rent for h in self.housing_stock])
        vacant_ratio = sum(1 for h in self.housing_stock if h.status == "vacant") / len(self.housing_stock)
        rental_ratio = sum(1 for h in self.housing_stock if h.status in ["for_rent", "rented"]) / len(
            self.housing_stock)

        displaced_count = len([a for a in self.agents if "displaced" in a and not self.agent_properties[a]["housed"]])
        displaced_ratio = displaced_count / max(1, len([a for a in self.agents if "displaced" in a]))

        # Government observation (full market view)
        if "government" in self.agents:
            gov_obs = np.array([
                self.current_step / self.max_steps,
                self.interest_rate,
                float(self.rent_control_active),
                avg_house_price / 100000,
                avg_rent / 1000,
                vacant_ratio,
                rental_ratio,
                displaced_ratio,
                len(self.housing_stock) / 100,
                sum(self.agent_properties[a]["wealth"] for a in self.agents if a != "government") / 1000000,
                sum(self.agent_properties[a]["income"] for a in self.agents if a != "government") / 100000,
                sum(len(self.agent_properties[a].get("owned_houses", [])) for a in self.agents if
                    "investor" in a) / len(self.housing_stock),
                sum(self.agent_properties[a].get("satisfaction", 0) for a in self.agents if "renter" in a) / max(1,
                                                                                                                 len([a
                                                                                                                      for
                                                                                                                      a
                                                                                                                      in
                                                                                                                      self.agents
                                                                                                                      if
                                                                                                                      "renter" in a])),
                sum(self.agent_properties[a].get("rental_income", 0) for a in self.agents if "investor" in a) / 10000,
                self.agent_properties["government"]["budget"] / 100000
            ], dtype=np.float32)
            observations["government"] = gov_obs

        # Individual agent observations
        for agent_id in self.agents:
            if agent_id == "government":
                continue

            agent_props = self.agent_properties[agent_id]
            agent_type = agent_props["type"]

            # Personal info
            wealth_norm = agent_props["wealth"] / 100000
            income_norm = agent_props["income"] / 10000
            housed = float(agent_props.get("housed", False))

            # Market info (limited view)
            affordable_houses = sum(1 for h in self.housing_stock
                                    if h.status == "vacant" and h.price <= agent_props["wealth"])
            affordable_rentals = sum(1 for h in self.housing_stock
                                     if h.status == "for_rent" and h.rent <= agent_props["income"] * 0.3)

            if agent_type == "displaced":
                obs = np.array([
                    wealth_norm, income_norm, housed,
                    affordable_houses / max(1, len(self.housing_stock)),
                    affordable_rentals / max(1, len(self.housing_stock)),
                    avg_house_price / 100000, avg_rent / 1000,
                    agent_props.get("satisfaction", 0) / 10,
                    self.interest_rate, float(self.rent_control_active)
                ], dtype=np.float32)

            elif agent_type == "renter":
                current_rent = agent_props.get("rent_payment", 0)
                obs = np.array([
                    wealth_norm, income_norm, housed,
                    current_rent / 1000, agent_props.get("satisfaction", 0) / 10,
                    affordable_houses / max(1, len(self.housing_stock)),
                    affordable_rentals / max(1, len(self.housing_stock)),
                    avg_house_price / 100000, avg_rent / 1000,
                    self.interest_rate
                ], dtype=np.float32)

            elif agent_type == "owner":
                equity = agent_props.get("equity", 0)
                num_owned = len(agent_props.get("owned_houses", []))
                obs = np.array([
                    wealth_norm, income_norm, equity / 100000,
                    num_owned, housed,
                                              avg_house_price / 100000, vacant_ratio,
                    rental_ratio, self.interest_rate,
                    float(self.rent_control_active)
                ], dtype=np.float32)

            elif agent_type == "investor":
                rental_income = agent_props.get("rental_income", 0)
                total_investment = agent_props.get("total_investment", 1)
                roi = rental_income / max(total_investment, 1) * 12  # Annualized ROI
                num_owned = len(agent_props.get("owned_houses", []))

                obs = np.array([
                    wealth_norm, income_norm, rental_income / 1000,
                    roi, num_owned,
                                              avg_house_price / 100000, avg_rent / 1000,
                    vacant_ratio, rental_ratio,
                    self.interest_rate
                ], dtype=np.float32)

            observations[agent_id] = obs

        return observations

    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all agents based on their objectives"""
        rewards = {}

        # Government reward: Market stability and affordability
        if "government" in self.agents:
            # Count ALL displaced agents (both original and converted renters)
            displaced_count = len([
                a for a in self.agents
                if self.agent_properties[a].get("type") == "displaced"
                   and not self.agent_properties[a]["housed"]
            ])

            displaced_penalty = -displaced_count * 10

            avg_satisfaction = np.mean([self.agent_properties[a].get("satisfaction", 0)
                                        for a in self.agents if "renter" in a or "displaced" in a])
            satisfaction_reward = avg_satisfaction * 5

            # Penalize extreme market conditions
            avg_house_price = np.mean([h.price for h in self.housing_stock])
            avg_income = np.mean([self.agent_properties[a]["income"] for a in self.agents if a != "government"])
            affordability_ratio = avg_house_price / avg_income

            if 3 <= affordability_ratio <= 5:
                affordability_reward = 20
            else:
                affordability_reward = -abs(affordability_ratio - 4) * 5

            rewards["government"] = displaced_penalty + satisfaction_reward + affordability_reward

        # Displaced rewards: Getting housed
        for agent_id in self.agents:
            if "displaced" in agent_id:
                agent_props = self.agent_properties[agent_id]
                if agent_props["housed"] and agent_props.get("house_id") is not None:
                    rewards[agent_id] = 100  # Big reward for getting housed
                elif agent_props["housed"]:
                    rewards[agent_id] = 50  # Smaller reward for renting
                else:
                    rewards[agent_id] = -10  # Penalty for remaining displaced

        # Renter rewards: Satisfaction with housing
        for agent_id in self.agents:
            if "renter" in agent_id:
                agent_props = self.agent_properties[agent_id]
                satisfaction = agent_props.get("satisfaction", 0)
                rent_burden = agent_props.get("rent_payment", 0) / agent_props["income"]

                if rent_burden < 0.3:  # Affordable housing
                    affordability_bonus = 20
                elif rent_burden > 0.5:  # Unaffordable
                    affordability_bonus = -30
                else:
                    affordability_bonus = 0

                rewards[agent_id] = satisfaction * 5 + affordability_bonus

        # Owner rewards: Equity growth and property value
        for agent_id in self.agents:
            if "owner" in agent_id:
                agent_props = self.agent_properties[agent_id]
                equity_reward = agent_props.get("equity", 0) / 10000  # Scale down

                # Bonus for owning multiple properties
                num_owned = len(agent_props.get("owned_houses", []))
                ownership_bonus = (num_owned - 1) * 10  # Bonus for additional properties

                rewards[agent_id] = equity_reward + ownership_bonus

        # Investor rewards: ROI and rental income
        for agent_id in self.agents:
            if "investor" in agent_id:
                agent_props = self.agent_properties[agent_id]
                rental_income = agent_props.get("rental_income", 0)
                total_investment = agent_props.get("total_investment", 1)

                # ROI reward (monthly rental income / total investment)
                roi = rental_income / max(total_investment, 1) * 100
                rewards[agent_id] = roi * 10  # Scale up for reward

        return rewards

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute actions for all agents"""
        self.current_step += 1

        # Complete construction projects
        completed_houses = []
        for project in self.construction_queue[:]:
            project["progress"] += 1
            if project["progress"] >= project["duration"]:
                completed_houses.append(project["house"])
                self.construction_queue.remove(project)

        for house in completed_houses:
            self.housing_stock.append(house)

        # Execute government actions first (they affect the market)
        if "government" in actions:
            self._execute_government_action(actions["government"])

        # Then execute individual agent actions
        for agent_id, action in actions.items():
            if agent_id != "government":
                self._execute_agent_action(agent_id, action)

        # Update agent states (income growth, satisfaction, etc.)
        self._update_agent_states()

        # Get new observations and rewards
        observations = self._get_observations()
        rewards = self._calculate_rewards()

        # Check termination
        terminated = {agent: self.current_step >= self.max_steps for agent in self.agents}
        truncated = {agent: False for agent in self.agents}

        # Info
        infos = {agent: {"step": self.current_step} for agent in self.agents}

        return observations, rewards, terminated, truncated, infos

    def _execute_government_action(self, action: int):
        """Execute government policy actions"""
        if action == 1:  # Lower interest rate
            self.interest_rate *= 0.95
            # Increase house prices due to cheaper credit
            for house in self.housing_stock:
                house.price *= 1.02
        elif action == 2:  # Raise interest rate
            self.interest_rate *= 1.05
            # Decrease house prices due to expensive credit
            for house in self.housing_stock:
                house.price *= 0.98
        elif action == 3:  # Build houses
            cost_per_house = 300000  # Realistic construction cost
            if self.agent_properties["government"]["budget"] >= cost_per_house:
                # Create house project with construction delay
                house = House(
                    self.house_price_avg * 1.2,  # More expensive to build than market value
                    self.house_price_std,
                    self.house_rent_avg * 0.7,  # Subsidized rent
                    self.house_rent_std
                )

                # Add to construction queue (3-6 month delay)
                self.construction_queue.append({
                    "house": house,
                    "start_step": self.current_step,
                    "duration": np.random.randint(5, 20),
                    "progress": 0
                })

                self.agent_properties["government"]["budget"] -= cost_per_house
        elif action == 4:  # Rent control on
            if not self.rent_control_active:
                self.rent_control_active = True
                for house in self.housing_stock:
                    house.rent *= 0.9
        elif action == 5:  # Rent control off
            if self.rent_control_active:
                self.rent_control_active = False
                for house in self.housing_stock:
                    house.rent *= 1.1

    def _execute_agent_action(self, agent_id: str, action: int):
        """Execute individual agent actions"""
        agent_props = self.agent_properties[agent_id]
        agent_type = agent_props["type"]

        if agent_type == "displaced":
            self._execute_displaced_action(agent_id, action)
        elif agent_type == "renter":
            self._execute_renter_action(agent_id, action)
        elif agent_type == "owner":
            self._execute_owner_action(agent_id, action)
        elif agent_type == "investor":
            self._execute_investor_action(agent_id, action)

    def _execute_displaced_action(self, agent_id: str, action: int):
        """Execute displaced agent actions"""
        agent_props = self.agent_properties[agent_id]

        if action == 1:  # Try to buy cheapest house
            affordable_houses = [h for h in self.housing_stock
                                 if h.status == "vacant" and h.price <= agent_props["wealth"]]
            if affordable_houses:
                cheapest = min(affordable_houses, key=lambda x: x.price)
                cheapest.status = "owned"
                cheapest.owner = agent_id
                agent_props["wealth"] -= cheapest.price
                agent_props["housed"] = True
                agent_props["house_id"] = cheapest.id
                agent_props["satisfaction"] = 8  # High satisfaction from owning

        elif action == 2:  # Try to rent cheapest
            affordable_rentals = [h for h in self.housing_stock
                                  if h.status == "for_rent" and h.rent <= agent_props["income"] * 0.4]
            if affordable_rentals:
                cheapest = min(affordable_rentals, key=lambda x: x.rent)
                cheapest.status = "rented"
                cheapest.tenant = agent_id
                agent_props["housed"] = True
                agent_props["house_id"] = cheapest.id
                agent_props["rent_payment"] = cheapest.rent
                agent_props["satisfaction"] = 5  # Medium satisfaction from renting

    def _execute_renter_action(self, agent_id: str, action: int):
        """Execute renter agent actions"""
        agent_props = self.agent_properties[agent_id]

        if action == 1:  # Move to cheaper place
            current_rent = agent_props.get("rent_payment", float('inf'))
            cheaper_rentals = [h for h in self.housing_stock
                               if h.status == "for_rent" and h.rent < current_rent]
            if cheaper_rentals:
                # Free up current house
                if agent_props.get("house_id"):
                    current_house = next((h for h in self.housing_stock if h.id == agent_props["house_id"]), None)
                    if current_house:
                        current_house.status = "for_rent"
                        current_house.tenant = None

                # Move to cheaper place
                new_house = min(cheaper_rentals, key=lambda x: x.rent)
                new_house.status = "rented"
                new_house.tenant = agent_id
                agent_props["house_id"] = new_house.id
                agent_props["rent_payment"] = new_house.rent
                agent_props["satisfaction"] += 2  # Happy about saving money

        elif action == 3:  # Try to buy
            if agent_props["wealth"] > 0:
                affordable_houses = [h for h in self.housing_stock
                                     if h.status == "vacant" and h.price <= agent_props["wealth"]]
                if affordable_houses:
                    house = min(affordable_houses, key=lambda x: x.price)
                    house.status = "owned"
                    house.owner = agent_id
                    agent_props["wealth"] -= house.price

                    # Free up rental
                    if agent_props.get("house_id"):
                        old_house = next((h for h in self.housing_stock if h.id == agent_props["house_id"]), None)
                        if old_house:
                            old_house.status = "for_rent"
                            old_house.tenant = None

                    agent_props["house_id"] = house.id
                    agent_props["rent_payment"] = 0
                    agent_props["satisfaction"] = 9  # Very happy about owning

                    # Convert to owner
                    agent_props["type"] = "owner"
                    agent_props["owned_houses"] = [house.id]
                    agent_props["equity"] = house.price

    def _execute_owner_action(self, agent_id: str, action: int):
        """Execute owner agent actions"""
        agent_props = self.agent_properties[agent_id]

        if action == 1:  # Sell house
            owned_houses = agent_props.get("owned_houses", [])
            if len(owned_houses) > 1:  # Don't sell primary residence
                house_to_sell = next((h for h in self.housing_stock if h.id == owned_houses[-1]), None)
                if house_to_sell:
                    house_to_sell.status = "vacant"
                    house_to_sell.owner = None
                    agent_props["wealth"] += house_to_sell.price
                    agent_props["owned_houses"].remove(house_to_sell.id)

        elif action == 2:  # Rent out house
            owned_houses = agent_props.get("owned_houses", [])
            for house_id in owned_houses:
                if house_id != agent_props.get("house_id"):  # Not primary residence
                    house = next((h for h in self.housing_stock if h.id == house_id), None)
                    if house and house.status == "owned":
                        house.status = "for_rent"

        elif action == 3:  # Buy another house
            affordable_houses = [h for h in self.housing_stock
                                 if h.status == "vacant" and h.price <= agent_props["wealth"]]
            if affordable_houses:
                house = min(affordable_houses, key=lambda x: x.price)
                house.status = "owned"
                house.owner = agent_id
                agent_props["wealth"] -= house.price
                agent_props["owned_houses"].append(house.id)
                agent_props["equity"] += house.price

    def _execute_investor_action(self, agent_id: str, action: int):
        """Execute investor agent actions"""
        agent_props = self.agent_properties[agent_id]

        if action == 1:  # Buy house for investment
            # Consider both vacant houses and owned houses where owner might sell
            potential_houses = [
                h for h in self.housing_stock
                if (h.status == "vacant" or
                    (h.status == "owned" and np.random.random() < 0.3))  # 30% chance owner will sell
                   and h.price <= agent_props["wealth"]
            ]

            if potential_houses:
                # Choose house with best rent-to-price ratio
                best_house = max(potential_houses, key=lambda x: x.rent / x.price)

                # Handle purchase from owner if needed
                if best_house.status == "owned":
                    # Transfer money to previous owner
                    prev_owner_id = best_house.owner
                    if prev_owner_id in self.agent_properties:
                        self.agent_properties[prev_owner_id]["wealth"] += best_house.price

                        # Remove house from previous owner's portfolio
                        if best_house.id in self.agent_properties[prev_owner_id].get("owned_houses", []):
                            self.agent_properties[prev_owner_id]["owned_houses"].remove(best_house.id)

                # Update house status and ownership
                best_house.status = "for_rent"
                best_house.owner = agent_id
                agent_props["wealth"] -= best_house.price
                agent_props["owned_houses"].append(best_house.id)
                agent_props["total_investment"] += best_house.price

        elif action == 2:  # Sell house
            owned_houses = agent_props.get("owned_houses", [])
            if owned_houses:
                house_to_sell = next((h for h in self.housing_stock if h.id == owned_houses[0]), None)
                if house_to_sell:
                    house_to_sell.status = "vacant"
                    house_to_sell.owner = None
                    if house_to_sell.tenant:
                        # Evict tenant
                        tenant_id = house_to_sell.tenant
                        if tenant_id in self.agent_properties:
                            self.agent_properties[tenant_id]["housed"] = False
                            self.agent_properties[tenant_id]["house_id"] = None
                            self.agent_properties[tenant_id]["satisfaction"] -= 3
                        house_to_sell.tenant = None

                    agent_props["wealth"] += house_to_sell.price
                    agent_props["owned_houses"].remove(house_to_sell.id)

        elif action == 3:  # Adjust rent on ALL vacant properties
            # Get ALL rental properties owned by this investor
            rental_properties = [
                h for h in self.housing_stock
                if h.owner == agent_id
                   and h.status in ["for_rent", "rented"]
            ]

            if not rental_properties:
                return  # No rental properties to adjust

            # Calculate market vacancy rate
            all_rentals = [h for h in self.housing_stock if h.status in ["for_rent", "rented"]]
            vacant_count = sum(1 for h in all_rentals if h.status == "for_rent")
            vacancy_rate = vacant_count / len(all_rentals) if all_rentals else 0.0

            # Adjust rent and check tenant affordability
            for property in rental_properties:
                old_rent = property.rent

                # Adjust based on market conditions
                if vacancy_rate < 0.1:  # Seller's market
                    property.rent *= 1.10
                elif vacancy_rate > 0.3:  # Buyer's market
                    property.rent *= 0.90

                # Apply minimum rent constraint
                min_rent = property.price * 0.005
                property.rent = max(property.rent, min_rent)

                if self.rent_control_active:
                    # Limit rent increase to 5% even in seller's market
                    property.rent = min(property.rent, old_rent * 1.05)

                # Check tenant affordability if property is occupied
                if property.status == "rented" and property.tenant:
                    tenant_id = property.tenant
                    tenant = self.agent_properties.get(tenant_id)

                    if tenant and property.rent > tenant["income"] * 0.5:
                        # Evict tenant who can't afford
                        tenant["housed"] = False
                        tenant["house_id"] = None
                        tenant["rent_payment"] = 0
                        tenant["satisfaction"] = max(0, tenant.get("satisfaction", 5) - 5)

                        # Update property status
                        property.status = "for_rent"
                        property.tenant = None

                # Update tenant's rent payment if they stay
                elif property.status == "rented" and property.tenant:
                    tenant_id = property.tenant
                    if tenant_id in self.agent_properties:
                        self.agent_properties[tenant_id]["rent_payment"] = property.rent

                # Log adjustment
                property.rent_history.append({
                    'step': self.current_step,
                    'old_rent': old_rent,
                    'new_rent': property.rent,
                    'vacancy_rate': vacancy_rate
                })

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        self.current_step = 0
        self.agents = self.possible_agents.copy()
        self.agent_properties = {}
        self.housing_stock = []
        self.interest_rate = 0.05
        self.rent_control_active = False

        # Reset House ID counter
        House.next_id = 1

        # Initialize agents and housing
        self._initialize_agents()
        self._create_housing_stock()

        # Reset history
        self.history = {
            'steps': [],
            'displaced_housed': [],
            'displaced_total': [],
            'renter_satisfaction': [],
            'owner_equity': [],
            'investor_returns': [],
            'market_stability': [],
            'people_displaced': [],
            'people_renters': [],
            'people_owners': [],
            'houses_owned_occupied': [],
            'houses_rented': [],
            'houses_vacant': [],
            'houses_for_rent': []
        }

        # Get initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def render(self):
        """Render the environment state."""
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            # Here you could add nice plots
            pass

    def final_render(self):
        """Render final area charts showing simulation trends with ownership breakdown"""
        plt.figure(figsize=(15, 12))

        # Household Types Over Time
        plt.subplot(3, 1, 1)
        plt.stackplot(
            self.history['steps'],
            self.history['people_displaced'],
            self.history['people_renters'],
            self.history['people_owners'],
            labels=['Displaced', 'Renters', 'Owners'],
            colors=['#ff7f0e', '#1f77b4', '#2ca02c']
        )
        plt.title('Household Composition Over Time')
        plt.ylabel('Number of Households')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        # Housing Status Over Time
        plt.subplot(3, 1, 2)
        plt.stackplot(
            self.history['steps'],
            self.history['houses_owned_occupied'],
            self.history['houses_rented'],
            self.history['houses_vacant'],
            self.history['houses_for_rent'],
            labels=['Owner-Occupied', 'Rented', 'Vacant (Available)', 'Vacant (Investor)'],
            colors=['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e']
        )
        plt.title('Housing Stock Status Over Time')
        plt.ylabel('Number of Houses')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        # Ownership Breakdown Over Time
        plt.subplot(3, 1, 3)

        # Calculate investor ownership percentage
        investor_owned = [
            (r + f) / total * 100 if total > 0 else 0
            for r, f, total in zip(
                self.history['houses_rented'],
                self.history['houses_for_rent'],
                [len(self.housing_stock)] * len(self.history['steps'])
            )
        ]

        owner_occupied_pct = [
            o / total * 100 if total > 0 else 0
            for o, total in zip(
                self.history['houses_owned_occupied'],
                [len(self.housing_stock)] * len(self.history['steps'])
            )
        ]

        plt.plot(
            self.history['steps'],
            investor_owned,
            label='Investor Owned (%)',
            color='#d62728',
            linewidth=2
        )

        plt.plot(
            self.history['steps'],
            owner_occupied_pct,
            label='Owner Occupied (%)',
            color='#2ca02c',
            linewidth=2
        )

        plt.title('Ownership Breakdown Over Time')
        plt.xlabel('Simulation Steps')
        plt.ylabel('Percentage of Housing Stock')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)

        plt.tight_layout()
        plt.show()


    def _update_agent_states(self):
        """Update agent states between steps (income, satisfaction, etc.)"""
        displaced_housed = 0
        renter_satisfaction = []
        owner_equity = []
        investor_returns = []

        # Initialize new metrics
        people_displaced = 0
        people_renters = 0
        people_owners = 0
        houses_owned_occupied = 0
        houses_rented = 0
        houses_vacant = 0
        houses_for_rent = 0

        # Count all displaced agents (both original and evicted renters)
        displaced_count = len([a for a in self.agents
                               if (self.agent_properties[a].get("type") in ["displaced", "renter"]
                                   and not self.agent_properties[a]["housed"])])

        # Update all agents
        for agent_id in self.agents:
            props = self.agent_properties[agent_id]
            agent_type = props["type"]

            # Monthly income accumulation
            if agent_type != "government":
                props["wealth"] += props["income"] / 12  # Monthly income

            # Type-specific updates
            if agent_type == "displaced":
                if props["housed"]:
                    displaced_housed += 1
                else:
                    # Degrade satisfaction for unhoused displaced
                    props["satisfaction"] = max(0, props.get("satisfaction", 5) - 1)

            elif agent_type == "renter":
                # Pay rent if housed
                if props["housed"] and props.get("rent_payment", 0) > 0:
                    props["wealth"] -= props["rent_payment"]
                    rent_burden = props["rent_payment"] / props["income"]

                    # Update satisfaction based on rent burden
                    if rent_burden > 0.5:
                        props["satisfaction"] = max(0, props["satisfaction"] - 2)
                    elif rent_burden > 0.3:
                        props["satisfaction"] = max(0, props["satisfaction"] - 1)
                    else:
                        props["satisfaction"] = min(10, props["satisfaction"] + 1)

                # Convert unhoused renters to displaced
                if not props["housed"]:
                    props["type"] = "displaced"
                    # Reset housing-related properties
                    props["house_id"] = None
                    props["rent_payment"] = 0
                    props["satisfaction"] = max(0, props.get("satisfaction", 5) - 3)

                renter_satisfaction.append(props.get("satisfaction", 5))

            elif agent_type == "owner":
                # Update equity based on current house prices
                equity = 0
                for house_id in props["owned_houses"]:
                    house = next(h for h in self.housing_stock if h.id == house_id)
                    equity += house.price
                props["equity"] = equity
                owner_equity.append(equity)

            elif agent_type == "investor":
                # Collect rental income
                rental_income = 0
                owned_houses = 0
                for house_id in props["owned_houses"]:
                    owned_houses += 1
                    house = next(h for h in self.housing_stock if h.id == house_id)
                    if house.status == "rented":
                        rental_income += house.rent
                        props["wealth"] += house.rent
                print(f"---> Owned houwses: {owned_houses}")
                props["rental_income"] = rental_income
                if props["total_investment"] > 0:
                    roi = (rental_income / props["total_investment"]) * 100
                    investor_returns.append(roi)

            if agent_type == "displaced":
                if not props["housed"]:
                    people_displaced += 1
            elif agent_type == "renter" and props["housed"]:
                people_renters += 1
            elif agent_type == "owner" and props["housed"]:
                people_owners += 1

        # Track housing status
        for house in self.housing_stock:
            if house.status == "owned":
                houses_owned_occupied += 1
            elif house.status == "rented":
                houses_rented += 1
            elif house.status == "vacant":
                houses_vacant += 1
            elif house.status == "for_rent":
                houses_for_rent += 1

        # Calculate market stability metric
        current_prices = [h.price for h in self.housing_stock]
        price_variance = np.var(current_prices) if current_prices else 0

        # Update history
        self.history['steps'].append(self.current_step)
        self.history['displaced_housed'].append(displaced_housed)
        self.history['displaced_total'].append(displaced_count)  # New metric
        self.history['renter_satisfaction'].append(np.mean(renter_satisfaction) if renter_satisfaction else 0)
        self.history['owner_equity'].append(np.mean(owner_equity) if owner_equity else 0)
        self.history['investor_returns'].append(np.mean(investor_returns) if investor_returns else 0)
        self.history['market_stability'].append(1 / (price_variance + 1e-6))

        self.history['people_displaced'].append(people_displaced)
        self.history['people_renters'].append(people_renters)
        self.history['people_owners'].append(people_owners)
        self.history['houses_owned_occupied'].append(houses_owned_occupied)
        self.history['houses_rented'].append(houses_rented)
        self.history['houses_vacant'].append(houses_vacant)
        self.history['houses_for_rent'].append(houses_for_rent)


def run_random_simulation():
    # Initialize environment with shorter episode length
    env = MultiAgentHousingEnv(
        max_steps=50,
        render_mode=None,  # Change to None to disable rendering
        num_displaced=40,
        num_renters=30,
        num_owners=22,
        num_investors=8,
        num_houses=60
    )

    observations, infos = env.reset()

    cumulative_rewards = {
        "government": 0.0,
        "displaced": 0.0,
        "renters": 0.0,
        "owners": 0.0,
        "investors": 0.0
    }

    for step in range(env.max_steps):
        # Generate random actions for all agents
        actions = {
            agent: env.action_spaces[agent].sample()
            for agent in env.agents
        }

        # Take environment step
        observations, rewards, terminated, truncated, infos = env.step(actions)

        # Update cumulative rewards
        for agent_id, reward in rewards.items():
            if agent_id == "government":
                cumulative_rewards["government"] += reward
            elif "displaced" in agent_id:
                cumulative_rewards["displaced"] += reward
            elif "renter" in agent_id:
                cumulative_rewards["renters"] += reward
            elif "owner" in agent_id:
                cumulative_rewards["owners"] += reward
            elif "investor" in agent_id:
                cumulative_rewards["investors"] += reward

        # Render current state
        if env.render_mode == "human":
            env.render()

        # Print step summary
        print(f"\nStep {step + 1}/{env.max_steps}")
        print(f"Government action: {actions['government']}")
        print(f"Market stats - Price: {np.mean([h.price for h in env.housing_stock]):.0f}, "
              f"Rent: {np.mean([h.rent for h in env.housing_stock]):.0f}")
        print(
            f"Housed displaced: {sum(1 for a in env.agents if 'displaced' in a and env.agent_properties[a]['housed'])}")
        print(f"Houses in Construction: {len(env.construction_queue)}")

    # Final report
    print("\n=== Simulation Complete ===")
    print("Cumulative Rewards:")
    for group, reward in cumulative_rewards.items():
        print(f"{group.capitalize():<12}: {reward:>8.1f}")

    print("\nFinal Market State:")
    print(f"Total houses: {len(env.housing_stock)}")
    print(f"Vacant: {sum(1 for h in env.housing_stock if h.status == 'vacant')}")
    print(f"Owned: {sum(1 for h in env.housing_stock if h.status == 'owned')}")
    print(f"Rented: {sum(1 for h in env.housing_stock if h.status == 'rented')}")
    print(f"Average owner equity: {np.mean(env.history['owner_equity'][-10:]):.0f}")
    print(f"Average investor ROI: {np.mean(env.history['investor_returns'][-10:]):.1f}%")

    env.final_render()


if __name__ == "__main__":
    run_random_simulation()