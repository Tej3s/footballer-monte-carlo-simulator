import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hard_em import HardEMJumpDiffusion, get_player_log_returns, get_player_info_on_date
from monte_carlo import FootballerMonteCarlo
from constrained_rls import ConstrainedRLS
from finding_priors import compute_contract_status
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, 'players.csv')):
    os.chdir(script_dir)
else:
    os.chdir(os.path.dirname(script_dir))
sys.path.append(os.getcwd())


st.set_page_config(page_title="Footballer Valuation Analysis", layout="wide")

st.title("Footballer Valuation Analysis")

@st.cache_data
def load_players():
    return pd.read_csv('players.csv')

def compute_years_left(contract_expiration_date, valuation_date):
    if pd.isna(contract_expiration_date):
        return None
    try:
        expiry_ts = pd.to_datetime(contract_expiration_date)
        return round((expiry_ts - pd.Timestamp(valuation_date)).days / 365.25, 2)
    except Exception:
        return None

def is_retired(player_row, valuation_date):
    last_season = player_row.get('last_season', None)
    if pd.isna(last_season):
        return False, None
    try:
        last_season = int(last_season)
        years_inactive = pd.Timestamp(valuation_date).year - last_season
        return years_inactive >= 2, years_inactive
    except (ValueError, TypeError):
        return False, None

@st.dialog("Player Retired")
def show_retired_dialog(player_name, last_season, years_inactive):
    st.error(
        f"**{player_name}** appears to be retired.\n\n"
        f"Last active season: **{last_season}** "
        f"({years_inactive} years before the valuation date).\n\n"
        f"Please select an active player."
    )
    if st.button("OK", use_container_width=True):
        st.rerun()


with st.sidebar:
    st.header("Player Selection")
     
    @st.cache_data
    def load_player_names():
        df = pd.read_csv('players.csv')
        return sorted(df['name'].dropna().unique().tolist())
    
    player_name = st.selectbox(
        "Search for a player",
        options=load_player_names(),
        index=None,
        placeholder="Type a name...",
    )
    
    # Resolve the player ID
    if player_name:
        players_df = pd.read_csv('players.csv')
        match = players_df[players_df['name'] == player_name]
        if len(match) > 0:
            player_row = match.iloc[0]
            player_id = int(match.iloc[0]['player_id'])
            st.caption(f"ID: {player_id}")
        else:
            
            
            player_row = None
            player_id = None
            st.error("Player not found")
    else:
        player_id = None
        player_row = None
    
    st.header("Analysis Parameters")
    
    valuation_date = st.date_input(
        "Valuation Date",
        value=pd.Timestamp.today(),
        help="The model uses valuations up to this date",
    )




    
    # Check if the player is retired
    if player_row is not None:
        retired, years_inactive = is_retired(player_row, valuation_date)
        if retired:
            show_retired_dialog(
                                last_season = player_row.get('last_season'), 
                                years_inactive = years_inactive,
                                player_name=player_row['name'],
                               )
            st.stop()

       

        years_left = compute_years_left(
            player_row.get('contract_expiration_date', None),
            valuation_date,
        )
        if years_left is not None:
            if years_left < 0:
                st.error(f" Contract expired {abs(years_left):.1f} years ago")
            elif years_left < 1.0:
                st.warning(f" Contract expires in {years_left:.1f} years — free-transfer regime")
            elif years_left < 2.0:
                st.info(f"Contract expires in {years_left:.1f} years — reduced leverage")
            else:
                st.success(f"Contract: {years_left:.1f} years left")
        else:
            st.caption("Contract expiry: unknown")

        today = pd.Timestamp.today().date()
        if valuation_date == today:

            contract_status = compute_contract_status(
                player_row.get('contract_expiration_date', None),
                valuation_date,
        )
        else:
            contract_status = None
            st.caption("Contract data only applies for today's date. "
                       "Historical valuations use position × age priors.")
    else:
        contract_status = None 

    # Auto-fill value and age from the data
    if player_id:
        auto_value, auto_age, auto_pos, auto_date = get_player_info_on_date(player_id, valuation_date)
        
        if auto_value is not None:
            st.caption(f"Valuation on {auto_date}: €{auto_value:,}")
            initial_value = st.number_input(
                "Current Value (€)",
                min_value=0,
                value=auto_value,
                step=1_000_000,
                help="Auto-filled from Transfermarkt. Override if needed.",
            )
        else:
            st.warning("No valuation found before this date")
            initial_value = st.number_input("Current Value (€)", min_value=0, value=50_000_000, step=1_000_000)
        
        if auto_age is not None:
            st.caption(f"Age on {auto_date}: {auto_age}")
            player_age = st.number_input(
                "Player Age",
                min_value=0,
                value=auto_age,
                step=1,
                help="Auto-filled from date of birth. Override if needed.",
            )
        else:
            player_age = st.number_input("Player Age", min_value=0, value=21, step=1)
        
        position = auto_pos if auto_pos else "Midfield"
        st.caption(f"Position: {position}")
    else:
        initial_value = st.number_input("Current Value (€)", min_value=0, value=80_000_000, step=1_000_000)
        player_age = st.number_input("Player Age", min_value=0, value=21, step=1)
        position = "Midfield"
    
    n_simulations = st.number_input("Number of Monte Carlo Simulations", min_value=1000, value=10000, step=1000)
    recent_years = st.slider("Recent Years for EM", min_value=1, max_value=4, value=2)
    
    # Auto-suggest horizon based on age
    if player_age >= 27:
        projection_years = 1
        st.info(f"Player is {player_age}. Projecting over 1 year (terminal decline regime).")
    else:
        projection_years = 2
        st.info(f"Player is {player_age}. Projecting over 2 years (development arc).")
    
    
    
    
    seed = st.number_input("Random Seed (optional)", min_value=None, value=42)
   

    run_button = st.button("Run Analysis")

if run_button:
    with st.spinner("Running analysis..."):
        player_data = get_player_log_returns(player_id=player_id, end_date = valuation_date)
        
        if player_data.empty:
            st.error(f"No data found for player ID: {player_id}")
            st.stop()

        log_returns = player_data.iloc[0]['log_returns']
        actual_name = player_data.iloc[0]['name']

        st.success(f"Found {actual_name} ({len(log_returns)} log-returns)")

        # Use the slider value for EM window
        recent_obs = recent_years * 3
        recent_log_returns = log_returns[-recent_obs:] if len(log_returns) >= recent_obs else log_returns

        if len(recent_log_returns) < 3:
            st.error(f"Not enough log returns (need at least 3)")
            st.stop()
        
        hard_em = HardEMJumpDiffusion(max_iter=50, kmax=1)
        results = hard_em.fit(recent_log_returns, dt=1.0, verbose=True)


        results['mu_original'] = results['mu']
        results['sigma_original'] = results['sigma']
            
         
        st.info(f"Using unweighted drift: μ = {results['mu']:.4f}")

        rls = ConstrainedRLS('prior_db.pkl')

        results = rls.estimate(
            log_returns=log_returns,
            position= position,       # ← ADD THIS
            age=player_age,
            contract_status=contract_status,
            current_value=initial_value,
            hard_em_results=results,
            verbose=False        
        )


        # Monte Carlo simulation (always 2 years)
        mc = FootballerMonteCarlo(
            player_name=actual_name,
            initial_value=initial_value,
            results=results,
            years= projection_years,  # Always 3 years
            time_steps_per_year=4,
            n_simulations=n_simulations,
            log_returns = log_returns,
            seed=seed
        )
        mc.run_simulations()

        final_values = mc.statistics['final_values']
        stats = {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std': np.std(final_values),
            'p5': np.percentile(final_values, 5),
            'p25': np.percentile(final_values, 25),
            'p75': np.percentile(final_values, 75),
            'p95': np.percentile(final_values, 95),
            'prob_increase': np.mean(final_values > initial_value),
            'prob_double': np.mean(final_values > 2 * initial_value),
            'prob_half': np.mean(final_values < 0.5 * initial_value),
        }

        #em parameters
        st.subheader("Jump-Diffusion Parameters")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("μ (Drift)", f"{results['mu']:.4f}")
        col2.metric("σ (Volatility)", f"{results['sigma']:.4f}")
        col3.metric("λ (Jump Intensity)", f"{results['lambda']:.4f}")
        col4.metric("μ_J (Jump Mean)", f"{results['mu_J']:.4f}")
        col5.metric("σ_J (Jump Volatility)", f"{results['sigma_J']:.4f}")
        st.caption(f"Observations: {results['n_observations']} | Jumps Detected: {results['n_jumps']} | Iterations: {results['iterations']}")
        st.caption(f"Valuation Date: {valuation_date.strftime('%Y-%m-%d')}")
        st.caption(f"Data Points: {len(log_returns)} | Last Valuation: {player_data.iloc[0]['last_date']}")

        #monte carlo results
        with st.expander(" Parameter Interpretation", expanded=False):
            st.markdown("""
            - **μ (Drift)**: The average expected return per time step.
            - **σ (Volatility)**: The standard deviation of returns, indicating risk.
            - **λ (Jump Intensity)**: The expected number of jumps per time step.
            - **μ_J (Jump Mean)**: The average size of jumps.
            - **σ_J (Jump Volatility)**: The variability of jump sizes.
            """)
        
        st.subheader("Monte Carlo Simulation Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Final Value", f"€{stats['mean']:,.2f}")
        col2.metric("Median Final Value", f"€{stats['median']:,.2f}")
        col3.metric("5th Percentile", f"€{stats['p5']:,.2f}")
        col4.metric("95th Percentile", f"€{stats['p95']:,.2f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Increase", f"{stats['prob_increase']*100:.2f}%")
        col2.metric("Probability of Doubling", f"{stats['prob_double']*100:.2f}%")
        col3.metric("Probability of Halving", f"{stats['prob_half']*100:.2f}%")

        
        # CHART 1: Valuation Paths with Confidence Intervals
   
        st.subheader("Valuation Paths with Confidence Intervals")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        n_sample = min(100, mc.n_simulations)
        sample_idx = np.random.choice(mc.n_simulations, n_sample, replace=False)
        for idx in sample_idx:
            ax1.plot(mc.statistics['times'], mc.simulations[idx] / 1e6, alpha=0.05, color='blue')
        ax1.fill_between(mc.statistics['times'], mc.statistics['p5'] / 1e6, mc.statistics['p95'] / 1e6, 
                         alpha=0.2, color='blue', label='5%-95%')
        ax1.fill_between(mc.statistics['times'], mc.statistics['p25'] / 1e6, mc.statistics['p75'] / 1e6, 
                         alpha=0.3, color='blue', label='25%-75%')
        ax1.plot(mc.statistics['times'], mc.statistics['mean'] / 1e6, 'r-', label='Mean', linewidth=2)
        ax1.axhline(y=initial_value / 1e6, color='gray', linestyle=':', label='Initial')
        ax1.set_xlabel('Weeks')
        ax1.set_ylabel('Market Value (€M)')
        ax1.set_title(f'{actual_name}: Valuation Paths')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        # CHART 2: Final Value Distribution
      
        st.subheader("Final Value Distribution")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(final_values / 1e6, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(initial_value / 1e6, color='gray', linestyle='--', label='Current Value')
        ax2.axvline(stats['mean'] / 1e6, color='green', linestyle='--', label=f"Mean: €{stats['mean']/1e6:.1f}M")
        ax2.axvline(stats['p5'] / 1e6, color='red', linestyle='--', label=f"5th Percentile: €{stats['p5']/1e6:.1f}M")
        ax2.axvline(stats['p95'] / 1e6, color='red', linestyle='--', label=f"95th Percentile: €{stats['p95']/1e6:.1f}M")
        ax2.set_xlabel('Future Value (€M)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        

     
        st.subheader("Probability of Exceeding Thresholds")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        times = mc.statistics['times']
        for thresh in thresholds:
            prob = np.mean(mc.simulations > initial_value * thresh, axis=0)
            ax4.plot(times, prob, label=f'>{thresh:.1f}x', linewidth=2)
        ax4.set_xlabel('Weeks')
        ax4.set_ylabel('Probability')
        ax4.set_title(f'{actual_name}: Probability of Exceeding Thresholds')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)



        st.download_button(
            label="Download Results (CSV)",
            data=pd.DataFrame({'final_values': final_values}).to_csv(index=False),
            file_name=f"{actual_name.replace(' ', '_')}_simulation_results.csv",
            mime="text/csv",
            key="download_results"
        )
