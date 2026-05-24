"""
Streamlit dashboard for the multi-agent housing market simulation.

Run with: streamlit run dashboard.py
"""

import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config
from utils import get_action, load_all_models, load_results, make_env

st.set_page_config(page_title="Housing Market RL", layout="wide")


def _init_session_state():
    defaults = {
        "models": None,
        "env": None,
        "history": [],
        "crisis_events": [],
        "running": False,
        "current_step": 0,
        "observations": None,
        "comparison_data": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_models_cached():
    if st.session_state["models"] is None:
        import logging
        logging.disable(logging.WARNING)
        st.session_state["models"] = load_all_models()
        logging.disable(logging.NOTSET)


def _model_status() -> dict:
    paths = {
        "displaced": config.PATHS["displaced_policy"] + ".zip",
        "renter": config.PATHS["renter_policy"] + ".zip",
        "owner": config.PATHS["owner_policy"] + ".zip",
        "investor": config.PATHS["investor_policy"] + ".zip",
        "government": config.PATHS["policymaker_policy"] + ".zip",
    }
    return {k: Path(v).exists() for k, v in paths.items()}


def _sidebar():
    st.sidebar.title("Configuration")

    preset = st.sidebar.selectbox(
        "City Preset", options=list(config.CITY_PRESETS.keys()), index=0
    )
    mode = st.sidebar.radio("Agent Mode", ["random", "heuristic", "trained"])

    preset_params = config.get_preset(preset)
    max_steps = st.sidebar.number_input(
        "Episode Length", min_value=10, max_value=500,
        value=preset_params.get("max_steps", 100),
    )
    seed = st.sidebar.number_input("Seed (0 = random)", min_value=0, value=42)
    seed = seed if seed > 0 else None

    speed = st.sidebar.slider("Playback delay (ms)", 0, 200, 50, step=10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Status")
    status = _model_status()
    for agent_type, exists in status.items():
        icon = "🟢" if exists else "🔴"
        st.sidebar.text(f"{icon} {agent_type}")

    if mode == "trained" and not any(status.values()):
        st.sidebar.warning("No trained models found. Will fall back to heuristic.")

    return preset, mode, max_steps, seed, speed


def _start_simulation(preset: str, mode: str, max_steps: int, seed: int | None):
    params = config.get_preset(preset)
    params.pop("interest_rate_start", None)
    params["max_steps"] = max_steps
    env = make_env(preset)
    env.max_steps = max_steps

    obs, _ = env.reset(seed=seed)

    st.session_state["env"] = env
    st.session_state["observations"] = obs
    st.session_state["history"] = []
    st.session_state["crisis_events"] = []
    st.session_state["current_step"] = 0
    st.session_state["running"] = True


def _step_simulation(mode: str):
    env = st.session_state["env"]
    observations = st.session_state["observations"]
    models = st.session_state["models"] or {}

    actions = {}
    for agent_id in env.agents:
        obs = observations.get(
            agent_id,
            np.zeros(env.observation_spaces[agent_id].shape, dtype=np.float32),
        )
        actions[agent_id] = get_action(
            agent_id, obs, models, mode, env.action_spaces[agent_id]
        )

    observations, rewards, terminated, truncated, infos = env.step(actions)

    st.session_state["observations"] = observations
    st.session_state["current_step"] = env.current_step

    if env.extended_history:
        st.session_state["history"] = list(env.extended_history)

    if all(terminated.values()):
        st.session_state["running"] = False


def _add_crisis_markers(fig: go.Figure, crisis_events: list):
    colors = {
        "recession": "red",
        "supply_shock": "orange",
        "migration_wave": "purple",
    }
    for event in crisis_events:
        fig.add_vline(
            x=event["step"],
            line_dash="dash",
            line_color=colors.get(event["type"], "gray"),
            annotation_text=event["type"].replace("_", " "),
            annotation_position="top",
            annotation_font_size=10,
        )


def _render_live_charts():
    history = st.session_state["history"]
    crisis_events = st.session_state["crisis_events"]

    if not history:
        st.info("Click 'Start Simulation' to begin.")
        return

    steps = [h["step"] for h in history]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps,
            y=[h["displacement_rate"] for h in history],
            mode="lines", name="Displacement Rate",
        ))
        fig.update_layout(title="Displacement Rate", height=300, margin=dict(t=40, b=30))
        _add_crisis_markers(fig, crisis_events)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps,
            y=[h["avg_rent_to_income"] for h in history],
            mode="lines", name="Avg Rent/Income",
        ))
        threshold = config.REWARD_WEIGHTS["government"]["rent_income_threshold"]
        fig.add_hline(y=threshold, line_dash="dot", line_color="red",
                      annotation_text=f"threshold ({threshold})")
        fig.update_layout(title="Rent-to-Income Ratio", height=300, margin=dict(t=40, b=30))
        _add_crisis_markers(fig, crisis_events)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps,
            y=[h["occupancy_fraction"] for h in history],
            mode="lines", name="Occupancy",
        ))
        fig.update_layout(title="Occupancy Fraction", height=300, margin=dict(t=40, b=30))
        _add_crisis_markers(fig, crisis_events)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=steps,
            y=[h["avg_house_price"] for h in history],
            mode="lines", name="Avg Price",
        ))
        fig.update_layout(title="Average House Price", height=300, margin=dict(t=40, b=30))
        _add_crisis_markers(fig, crisis_events)
        st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    reward_types = ["displaced", "renter", "owner", "investor", "government"]
    for rtype in reward_types:
        fig.add_trace(go.Scatter(
            x=steps,
            y=[h["rewards_by_type"].get(rtype, 0.0) for h in history],
            mode="lines", name=rtype,
        ))
    fig.update_layout(title="Mean Rewards by Agent Type", height=350, margin=dict(t=40, b=30))
    _add_crisis_markers(fig, crisis_events)
    st.plotly_chart(fig, use_container_width=True)


def _render_crisis_panel():
    st.subheader("Crisis Injection")
    env = st.session_state["env"]
    if env is None:
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Trigger Recession", disabled=not st.session_state["running"]):
            env.trigger_recession()
            st.session_state["crisis_events"].append(
                {"step": env.current_step, "type": "recession"}
            )

    with col2:
        if st.button("Trigger Supply Shock", disabled=not st.session_state["running"]):
            env.trigger_supply_shock()
            st.session_state["crisis_events"].append(
                {"step": env.current_step, "type": "supply_shock"}
            )

    with col3:
        if st.button("Trigger Migration Wave", disabled=not st.session_state["running"]):
            env.trigger_migration_wave()
            st.session_state["crisis_events"].append(
                {"step": env.current_step, "type": "migration_wave"}
            )


def _scan_result_files() -> dict:
    """Scan results directory for all JSON files matching scenario convention."""
    results_dir = Path(config.PATHS["results_dir"])
    if not results_dir.exists():
        return {}

    all_results = {}
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            data = load_results(str(json_file))
            if "scenarios" in data:
                all_results[json_file.stem] = data
        except (ValueError, KeyError):
            continue

    return all_results


def _render_comparison_panel():
    st.header("Scenario Comparison")

    result_files = _scan_result_files()

    if not result_files:
        st.info(
            "No evaluation results found. Run evaluate.py first to generate comparison data."
        )
        return

    file_names = list(result_files.keys())
    selected_file = st.selectbox("Result File", file_names)
    data = result_files[selected_file]

    scenarios = data.get("scenarios", {})
    if not scenarios:
        st.warning("Selected file has no scenario data.")
        return

    scenario_names = list(scenarios.keys())

    table_rows = []
    for name in scenario_names:
        agg = scenarios[name].get("aggregate", {})
        row = {
            "Scenario": name,
            "Displacement Rate": agg.get("mean_mean_displacement_rate", 0.0),
            "Rent/Income": agg.get("mean_mean_rent_to_income", 0.0),
            "Occupancy": agg.get("mean_final_occupancy", 0.0),
            "Price Volatility": agg.get("mean_price_volatility", 0.0),
        }
        mean_rewards = agg.get("mean_rewards", {})
        for rtype in ["displaced", "renter", "owner", "investor", "government"]:
            row[f"Reward ({rtype})"] = mean_rewards.get(rtype, 0.0)
        table_rows.append(row)

    st.dataframe(table_rows, use_container_width=True)

    st.subheader("Displacement Rate Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=[scenarios[n]["aggregate"].get("mean_mean_displacement_rate", 0) for n in scenario_names],
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][:len(scenario_names)],
    ))
    fig.update_layout(height=350, margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Metric Over Time (Averaged Across Episodes)")
    metric_options = [
        "displacement_rate", "avg_rent_to_income", "occupancy_fraction",
        "avg_house_price", "informal_activity", "price_trend",
    ]
    selected_metric = st.selectbox("Metric", metric_options)

    fig = go.Figure()
    for name in scenario_names:
        episodes = scenarios[name].get("episodes", [])
        if not episodes:
            continue

        max_len = max(len(ep.get("steps", [])) for ep in episodes)
        if max_len == 0:
            continue

        metric_matrix = []
        for ep in episodes:
            values = [s.get(selected_metric, 0.0) for s in ep.get("steps", [])]
            if len(values) < max_len:
                values.extend([values[-1]] * (max_len - len(values)))
            metric_matrix.append(values)

        metric_array = np.array(metric_matrix)
        mean_vals = np.mean(metric_array, axis=0)
        std_vals = np.std(metric_array, axis=0)
        x = list(range(1, max_len + 1))

        fig.add_trace(go.Scatter(
            x=x, y=mean_vals.tolist(),
            mode="lines", name=name,
        ))
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=(mean_vals + std_vals).tolist() + (mean_vals - std_vals).tolist()[::-1],
            fill="toself",
            fillcolor="rgba(100,100,100,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name=f"{name} (std)",
        ))

    fig.update_layout(
        title=f"{selected_metric} Over Time",
        height=400, margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crisis Impact (Stressed - Baseline)")
    modes = set()
    for name in scenario_names:
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("baseline", "stressed"):
            modes.add(parts[0])

    if modes:
        impact_rows = []
        for mode in sorted(modes):
            baseline_key = f"{mode}_baseline"
            stressed_key = f"{mode}_stressed"
            if baseline_key in scenarios and stressed_key in scenarios:
                b_agg = scenarios[baseline_key].get("aggregate", {})
                s_agg = scenarios[stressed_key].get("aggregate", {})
                impact_rows.append({
                    "Mode": mode,
                    "Displacement Delta": (
                        s_agg.get("mean_mean_displacement_rate", 0)
                        - b_agg.get("mean_mean_displacement_rate", 0)
                    ),
                    "Rent/Income Delta": (
                        s_agg.get("mean_mean_rent_to_income", 0)
                        - b_agg.get("mean_mean_rent_to_income", 0)
                    ),
                    "Occupancy Delta": (
                        s_agg.get("mean_final_occupancy", 0)
                        - b_agg.get("mean_final_occupancy", 0)
                    ),
                    "Price Vol Delta": (
                        s_agg.get("mean_price_volatility", 0)
                        - b_agg.get("mean_price_volatility", 0)
                    ),
                })
        if impact_rows:
            st.dataframe(impact_rows, use_container_width=True)
        else:
            st.info("Need both baseline and stressed runs to compute impact.")
    else:
        st.info("No paired baseline/stressed scenarios found.")


def main():
    _init_session_state()
    _load_models_cached()

    preset, mode, max_steps, seed, speed = _sidebar()

    tab_live, tab_compare = st.tabs(["Live Simulation", "Compare Scenarios"])

    with tab_live:
        st.header("Live Simulation")

        col_start, col_stop, col_status = st.columns([1, 1, 3])
        with col_start:
            if st.button("Start Simulation"):
                _start_simulation(preset, mode, max_steps, seed)
        with col_stop:
            if st.button("Stop"):
                st.session_state["running"] = False
        with col_status:
            if st.session_state["running"]:
                step = st.session_state["current_step"]
                st.success(f"Running — step {step}/{max_steps}")
            elif st.session_state["history"]:
                st.info(f"Completed — {len(st.session_state['history'])} steps recorded.")

        _render_crisis_panel()
        _render_live_charts()

        if st.session_state["running"]:
            _step_simulation(mode)
            time.sleep(speed / 1000.0)
            st.rerun()

    with tab_compare:
        _render_comparison_panel()


if __name__ == "__main__":
    main()
