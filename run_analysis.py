

# run_analysis.py
from constrained_rls import ConstrainedRLS
from hard_em import HardEMJumpDiffusion,calculate_weighted_drift, get_player_log_returns
from monte_carlo import FootballerMonteCarlo

from typing import Optional
import numpy as np
import pandas as pd

def prepare_historical_data():

    players_df = pd.read_csv('players.csv')
    valuations_df = pd.read_csv('player_valuations.csv')
    
    historical_data = valuations_df.merge(
        players_df[['player_id', 'position', 'date_of_birth']],
        on='player_id',
        how='left'
    )
    
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data['date_of_birth'] = pd.to_datetime(historical_data['date_of_birth'])
    historical_data['age'] = (historical_data['date'] - historical_data['date_of_birth']).dt.days / 365.25
    
    return historical_data

def run_complete_analysis(player_name: str, 
                          initial_value: float, 
                          log_returns: np.ndarray,
                          player_age: Optional[int] = None,
                          use_weighted_drift_for_mc: bool = True,
                          position: Optional[str] = None,
                          use_recent_em: bool = True,
                          recent_years: int = 4,
                          half_life: int = 12,
                          n_simulations: int = 10000,
                          years: int = 3,
                          use_mean_reversion: bool = True,
                          historical_data: Optional[pd.DataFrame] = None,
                          seed: Optional[int] = 42):  # NEW 
    
    print(f"\n{'='*60}")
    print(f"COMPLETE ANALYSIS: {player_name}")
    print(f"{'='*60}")
    
    # Step 1: Run Hard EM (unweighted - estimates full career)
    print(f"\n Step 1: Running Hard EM...")

    recent_years = 3
    recent_obs = recent_years * 4 # Assuming weekly data
    recent_log_returns = log_returns[-recent_obs:] if len(log_returns) >= recent_obs else log_returns



    hard_em = HardEMJumpDiffusion(max_iter=50, kmax=1)
    results = hard_em.fit(recent_log_returns, dt=1.0, verbose=True)
    
    # Step 2: Adjust drift for Monte Carlo (if needed)
    if use_weighted_drift_for_mc and player_age is not None and player_age >= 30:
        print(f"\n Step 1b: Adjusting drift for Monte Carlo...")
        
        # Calculate weighted drift
        weighted = calculate_weighted_drift(log_returns, half_life=half_life)
        
        # Store original for reference
        results['mu_original'] = results['mu']
        results['sigma_original'] = results['sigma']
        
        # Override with weighted values for simulation
        results['mu'] = weighted['weighted_mu']
        results['sigma'] = weighted['weighted_sigma']
        results['mu_weighted'] = weighted['weighted_mu']
        results['sigma_weighted'] = weighted['weighted_sigma']
        results['half_life'] = half_life
        
        print(f"      Original μ: {results['mu_original']:.4f} (full career)")
        print(f"      Weighted μ: {results['mu']:.4f} (used for simulation)")
        print(f"      Original σ: {results['sigma_original']:.4f}")
        print(f"      Weighted σ: {results['sigma']:.4f}")
    else:
        print(f"\n Step 1b: Using unweighted drift for simulation")
        print(f"      μ = {results['mu']:.4f} (full career)")

    ceiling = None
    mu_decay = None
    jump_decay = None


    
    # Step 3: Monte Carlo (uses modified results)
    rls = ConstrainedRLS('prior_db.pkl')
    results = rls.estimate(
        log_returns=log_returns,
        position=position,
      
        current_value=initial_value,
        hard_em_results=results,
        verbose=False
)



    mc = FootballerMonteCarlo(
        player_name=player_name,
        
        initial_value=initial_value,
        results=results,  # ← This now has adjusted mu for veterans
        years=years,
        time_steps_per_year=4,  # Quarterly
        n_simulations=n_simulations,
        log_returns = log_returns,
       

        seed=seed
    )
    mc.run_simulations()
    mc.plot_simulations(save_path=f'{player_name.lower().replace(" ", "_")}_mc.png')
    
    # Step 4: Risk metrics
    risk_95 = mc.get_value_at_risk(0.95)
    risk_99 = mc.get_value_at_risk(0.99)
    
    print(f"\n Risk Metrics:")
    print(f"   {'Metric':<30} {'Value':>15}")
    print(f"   {'-'*46}")
    print(f"   {'VaR (95%)':<30} {'€{:,.0f}':>15}".format(risk_95['VaR']))
    print(f"   {'CVaR (95%)':<30} {'€{:,.0f}':>15}".format(risk_95['CVaR']))
    print(f"   {'Upside (95%)':<30} {'€{:,.0f}':>15}".format(risk_95['Upside']))
    print(f"   {'VaR (99%)':<30} {'€{:,.0f}':>15}".format(risk_99['VaR']))
    
    return results, mc


# In main():
def main():
    players_df = pd.read_csv('players.csv')
    valuations_df = pd.read_csv('player_valuations.csv')    

    historical_data = prepare_historical_data()
    print(f"   Prepared {len(historical_data)} historical records")

    players = [
        (861410, "Arda Guler", 15_000_000, 18),
        (683840, "Pedri", 80_000_000, 21),       # Young - keep EM mu
        (38253, "Robert Lewandowski", 25_000_000, 35),  # Veteran - use weighted
        (434675, "Cody Gakpo", 50_000_000, 24),    # Young - keep EM mu
        (500689, "Evann Guessand", 5_000_000, 22),     # Young - keep EM mu
        (566723, "Michael Olise", 50_000_000, 22),
        (937958, "Lamine Yamal", 60_000_000, 16),
        (573775, "Hugo Duro", 8_000_000, 24),
        (543499, "Mohammed Kudus", 40_000_000, 22),
        (435772, "Charles De Ketelaere", 25_000_000, 22),
        (709726, "Hugo Ekitike", 15_000_000, 21),
        (743591, "Sávio", 30_000_000, 19),
        (8198, "Cristiano Ronaldo", 20_000_000, 37),
        (451276, "Dominik Szoboszlai", 50_000_000, 22),
    ]
    
    for player_id, name, value, age in players:
        player_data = get_player_log_returns(player_id=player_id)
        log_returns = player_data.iloc[0]['log_returns']
        
        # For veterans, use weighted drift in Monte Carlo
        use_weighted = age >= 30
        
        results, mc = run_complete_analysis(
            player_name=name,
            initial_value=value,
            log_returns=log_returns,
            player_age=age,
            use_weighted_drift_for_mc=use_weighted,  # ← Only for veterans
            half_life=12,
            n_simulations=10000,
            
        )

if __name__ == "__main__":
    main()