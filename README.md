# Multi-Agent Reinforcement Learning Housing Market Simulation

A multi-agent RL simulation of urban housing markets where displaced persons, renters, owners, investors, and a government policy maker interact under configurable economic conditions. Agents are trained with PPO via stable-baselines3 and evaluated against heuristic and random baselines under crisis scenarios.

## File Descriptions

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters, reward weights, city presets, training settings, and paths — no hardcoded values elsewhere. |
| `agents.py` | Observation extensions, observation slices, five reward functions, and five heuristic policy classes. |
| `environment.py` | Extended PettingZoo environment subclassing the starter code with salary growth, seasonal demand, informal markets, crisis injection, and dormant agent pool. |
| `train.py` | Two-stage PPO training pipeline — Stage 1 trains household agents against heuristics, Stage 2 trains the policy maker against frozen household policies. |
| `evaluate.py` | Experiment runner that executes random/heuristic/trained scenarios in baseline and stressed conditions, saving results to JSON. |
| `dashboard.py` | Streamlit interactive dashboard with live step-by-step simulation, crisis injection buttons, and scenario comparison charts. |
| `utils.py` | Shared helpers for environment creation, model loading, episode execution, metrics computation, and JSON serialization. |
| `problemA_starter_code.py` | Original starter code providing `MultiAgentHousingEnv` and the `House` class — not modified. |
| `requirements.txt` | Pinned Python dependencies. |

## Setup

1. Clone the repository and enter the directory:
   ```bash
   cd housing-market-rl
   ```

2. Create and activate a Python 3.11 virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training

Run the full two-stage training pipeline:
```bash
python train.py
```

Options:
```bash
python train.py --stage1-only          # Train household agents only
python train.py --stage2-only          # Train policy maker only (requires Stage 1 models)
python train.py --preset madrid        # Use Madrid calibration preset
python train.py --seed 42              # Set random seed
```

Trained models are saved as `.zip` files in the `models/` directory.

## Evaluation

Run all scenarios (random, heuristic, trained) in baseline and stressed conditions:
```bash
python evaluate.py --episodes 10
```

Options:
```bash
python evaluate.py --scenario heuristic    # Run only heuristic scenario
python evaluate.py --no-crisis             # Skip stressed conditions
python evaluate.py --preset madrid         # Use Madrid preset
python evaluate.py --seed 42
```

Results are saved as JSON in the `results/` directory.

## Dashboard

Launch the interactive Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The dashboard provides:
- Live step-by-step simulation with configurable agent mode and city preset
- Real-time crisis injection buttons (recession, supply shock, migration wave)
- Scenario comparison panel loading evaluation results from the `results/` directory

## Madrid Calibration Sources

The `madrid` city preset in `config.py` uses values calibrated from official European statistical sources:

| Parameter | Value | Source |
|-----------|-------|--------|
| `income_avg` | 30,000 | INE Spain Encuesta de Estructura Salarial 2022 |
| `wealth_avg` | 150,000 | Banco de España Encuesta Financiera de las Familias 2020 |
| `house_price_avg` | 350,000 | INE Spain Índice de Precios de Vivienda Q4 2023 |
| `house_rent_avg` | 1,400 | Eurostat Housing Cost Statistics 2023 |
| `interest_rate_start` | 0.045 | ECB Statistical Data Warehouse Q4 2023 |
| `num_owners` | 35 | Eurostat Housing Statistics 2023 (Spain homeownership rate 75%) |
| `num_displaced` | 15 | INE Spain Encuesta de Condiciones de Vida 2023 |
| `num_houses` | 70 | Madrid vacancy rate 1-2% (Eurostat) |
