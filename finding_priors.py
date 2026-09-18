import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Optional
from hard_em import get_player_log_returns, HardEMJumpDiffusion


def build_prior_database(min_valuations: int = 15, verbose: bool = True) -> Dict:
    
    if verbose:
        print(f"\n{'='*60}")
        print("BUILDING PRIOR DATABASE")
        print(f"{'='*60}")
    
    # Step 1: Load data
    if verbose:
        print("Loading player data via get_player_log_returns()...")
    
    all_players = get_player_log_returns(min_valuations=min_valuations)
    
    if all_players is None or len(all_players) == 0:
        print("No player data found!")
        return {}
    
    if verbose:
        print(f"Loaded {len(all_players)} players")
        print(f"Columns: {all_players.columns.tolist()}")
    
    # Step 2: Check observation column
    if 'n_valuations' in all_players.columns:
        obs_col = 'n_valuations'
    elif 'n_log_returns' in all_players.columns:
        obs_col = 'n_log_returns'
    else:
        print(" No observation column found!")
        return {}
    
    if verbose:
        print(f"Using '{obs_col}' for observations")
    
    # Step 3: Filter players with enough data
    good_players = all_players[all_players[obs_col] >= min_valuations]
    
    if verbose:
        print(f"Players with {min_valuations}+ observations: {len(good_players)}")
    
    if len(good_players) == 0:
        print(f"No players with {min_valuations}+ observations!")
        return {}
    
    # Step 4: Check if we have age data
    has_age = 'age' in good_players.columns
    if verbose:
        print(f"Age data available: {has_age}")
        if has_age:
            print(f"Age range: {good_players['age'].min():.1f} - {good_players['age'].max():.1f}")
    
    # Step 5: Build priors
    prior_db = {}
    
    # Age groups (if we have age data)
    age_groups = {
        'young': (0, 22),
        'prime': (22, 27),
        'veteran': (27, 35),
        'aged': (35, 99)
    }
    
    for position in good_players['position'].unique():
        if verbose:
            print(f"\n  Processing position: {position}")
        
        pos_players = good_players[good_players['position'] == position]
        
        if has_age:
            # Build priors by age group
            for age_group, (age_min, age_max) in age_groups.items():
                age_players = pos_players[
                    (pos_players['age'] >= age_min) & 
                    (pos_players['age'] < age_max)
                ]
                
                if len(age_players) >= 10:
                    if verbose:
                        print(f"    {age_group}: {len(age_players)} players")
                    prior_db[(position, age_group)] = calculate_priors(age_players)
                else:
                    if verbose and len(age_players) > 0:
                        print(f"    {age_group}: {len(age_players)} players (skipping, need 10)")
            
            # Also build position-only priors as fallback
            if len(pos_players) >= 10:
                if verbose:
                    print(f"    position-only: {len(pos_players)} players (fallback)")
                prior_db[position] = calculate_priors(pos_players)
        else:
            # No age data: position-only priors
            if len(pos_players) >= 10:
                if verbose:
                    print(f"    position-only: {len(pos_players)} players")
                prior_db[position] = calculate_priors(pos_players)
    
    # Step 6: Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"PRIOR DATABASE COMPLETE: {len(prior_db)} groups")
        print(f"{'='*60}")
        
        for key in prior_db.keys():
            if isinstance(key, tuple):
                pos, age = key
                label = f"{pos}, {age}"
            else:
                label = key
            print(f"  {label}: {prior_db[key]['n_players']} players")
    
    return prior_db


def calculate_priors(players: pd.DataFrame) -> Dict:
    """
    Calculate priors from a group of players.
    Uses Hard EM to detect jumps (asymmetric, sign-conditional).
    """
    mu_values = players['mu'].dropna()
    sigma_values = players['sigma'].dropna()

    # --- Robust filtering for mu ---
    mu_q1 = np.percentile(mu_values, 25)
    mu_q3 = np.percentile(mu_values, 75)
    mu_iqr = mu_q3 - mu_q1
    mu_lower = mu_q1 - 1.5 * mu_iqr
    mu_upper = mu_q3 + 1.5 * mu_iqr
    mu_filtered = mu_values[(mu_values >= mu_lower) & (mu_values <= mu_upper)]

    # --- Robust filtering for sigma ---
    sigma_q1 = np.percentile(sigma_values, 25)
    sigma_q3 = np.percentile(sigma_values, 75)
    sigma_iqr = sigma_q3 - sigma_q1
    sigma_lower = sigma_q1 - 1.5 * sigma_iqr
    sigma_upper = sigma_q3 + 1.5 * sigma_iqr
    sigma_filtered = sigma_values[(sigma_values >= sigma_lower) & (sigma_values <= sigma_upper)]

    # --- Jump parameters via Hard EM ---
    hem = HardEMJumpDiffusion(max_iter=50, kmax=1)

    lambda_values = []
    mu_j_values = []
    sigma_j_values = []
    lambda_up_values = []
    lambda_down_values = []
    mu_j_up_values = []
    mu_j_down_values = []
    sigma_j_up_values = []
    sigma_j_down_values = []

    for idx, row in players.iterrows():
        log_returns = row.get('log_returns', [])
        if not isinstance(log_returns, (list, np.ndarray)):
            continue
        if len(log_returns) < 5:
            continue

        returns = np.array(log_returns, dtype=float)

        try:
            res = hem.fit(returns, dt=1.0, verbose=False)
        except Exception:
            continue

        lambda_values.append(res['lambda'])
        mu_j_values.append(res['mu_J'])
        sigma_j_values.append(res['sigma_J'])

        if res['n_jumps'] > 0:
            lambda_up_values.append(res['lambda_up'])
            lambda_down_values.append(res['lambda_down'])
            mu_j_up_values.append(res['mu_J_up'])
            mu_j_down_values.append(res['mu_J_down'])
            sigma_j_up_values.append(res['sigma_J_up'])
            sigma_j_down_values.append(res['sigma_J_down'])

    if not lambda_values:
        lambda_values = [0.08]
        mu_j_values = [0.0]
        sigma_j_values = [0.10]

    if not lambda_up_values:
        lambda_up_values = [0.01]
        lambda_down_values = [0.01]
        mu_j_up_values = [0.30]
        mu_j_down_values = [-0.30]
        sigma_j_up_values = [0.15]
        sigma_j_down_values = [0.15]

    # Only use non-zero jumps for mu_J
    non_zero_mu_j = [x for x in mu_j_values if abs(x) > 0.01]
    non_zero_sigma_j = [x for x in sigma_j_values if x > 0.01]

    if non_zero_mu_j:
        mu_j_median = np.median(non_zero_mu_j)
        mu_j_mean = np.mean(non_zero_mu_j)
        mu_j_std = np.std(non_zero_mu_j)
        if len(non_zero_mu_j) < 5:
            print(f"    WARNING: only {len(non_zero_mu_j)} players had jumps")
        else:
            print(f"    {len(non_zero_mu_j)} players had jumps")
        print(f"    μ_J mean = {mu_j_mean:.4f}, median = {mu_j_median:.4f}")
    else:
        mu_j_median = 0.0
        mu_j_mean = 0.0
        mu_j_std = 0.05

    if non_zero_sigma_j:
        sigma_j_median = np.median(non_zero_sigma_j)
        sigma_j_mean = np.mean(non_zero_sigma_j)
        sigma_j_std = np.std(non_zero_sigma_j)
    else:
        sigma_j_median = 0.10
        sigma_j_mean = 0.10
        sigma_j_std = 0.04

    # Split means
    mu_j_up_mean = np.mean(mu_j_up_values)
    mu_j_up_std = np.std(mu_j_up_values)
    mu_j_up_median = np.median(mu_j_up_values)

    mu_j_down_mean = np.mean(mu_j_down_values)
    mu_j_down_std = np.std(mu_j_down_values)
    mu_j_down_median = np.median(mu_j_down_values)

    sigma_j_up_mean = np.mean(sigma_j_up_values)
    sigma_j_up_std = np.std(sigma_j_up_values)
    sigma_j_up_median = np.median(sigma_j_up_values)

    sigma_j_down_mean = np.mean(sigma_j_down_values)
    sigma_j_down_std = np.std(sigma_j_down_values)
    sigma_j_down_median = np.median(sigma_j_down_values)

    lambda_up_mean = np.mean(lambda_up_values)
    lambda_down_mean = np.mean(lambda_down_values)

    return {
        'mu': {
            'mean': np.mean(mu_filtered),
            'std': np.std(mu_filtered),
            'median': np.median(mu_filtered),
            'p5': np.percentile(mu_filtered, 5),
            'p95': np.percentile(mu_filtered, 95)
        },
        'sigma': {
            'mean': np.mean(sigma_filtered),
            'std': np.std(sigma_filtered),
            'median': np.median(sigma_filtered),
            'p5': np.percentile(sigma_filtered, 5),
            'p95': np.percentile(sigma_filtered, 95)
        },
        'lambda': {
            'mean': np.mean(lambda_values),
            'std': np.std(lambda_values),
            'median': np.median(lambda_values)
        },
        'mu_J': {
            'mean': mu_j_mean,
            'std': mu_j_std,
            'median': mu_j_median
        },
        'sigma_J': {
            'mean': sigma_j_mean,
            'std': sigma_j_std,
            'median': sigma_j_median
        },

        'mu_J_up': {
            'mean': mu_j_up_mean,
            'std': mu_j_up_std,
            'median': mu_j_up_median
        },
        'mu_J_down': {
            'mean': mu_j_down_mean,
            'std': mu_j_down_std,
            'median': mu_j_down_median
        },
        'sigma_J_up': {
            'mean': sigma_j_up_mean,
            'std': sigma_j_up_std,
            'median': sigma_j_up_median
        },
        'sigma_J_down': {
            'mean': sigma_j_down_mean,
            'std': sigma_j_down_std,
            'median': sigma_j_down_median
        },
        'lambda_up': {
            'mean': lambda_up_mean
        },
        'lambda_down': {
            'mean': lambda_down_mean
        },

        'n_players': len(players)
    }


def save_prior_database(prior_db: Dict, filepath: str = 'prior_db.pkl') -> None:
    """Save prior database to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(prior_db, f)
    print(f"Prior database saved to {filepath}")


def load_prior_database(filepath: str = 'prior_db.pkl') -> Dict:
    """Load prior database from disk."""
    if not os.path.exists(filepath):
        print(f"  Prior database not found at {filepath}")
        return {}
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def inspect_prior_database(filepath: str = 'prior_db.pkl'):
    """Load and display the prior database in a readable format."""
    
    prior_db = load_prior_database(filepath)
    
    if not prior_db:
        print(" No prior database found!")
        return
    
    print(f"\n{'='*70}")
    print(f"PRIOR DATABASE INSPECTION")
    print(f"{'='*70}")
    print(f"Total groups: {len(prior_db)}")
    
    position_groups = {}
    age_groups = {}
    
    for key, priors in prior_db.items():
        if isinstance(key, tuple):
            pos, age = key
            age_groups[(pos, age)] = priors
        else:
            position_groups[key] = priors
    
    if position_groups:
        print(f"\n POSITION-ONLY PRIORS (fallback):")
        print("-" * 50)
        for position, priors in position_groups.items():
            print(f"\n  {position}:")
            print(f"    μ: {priors['mu']['mean']:.4f} ± {priors['mu']['std']:.4f}")
            print(f"    σ: {priors['sigma']['mean']:.4f} ± {priors['sigma']['std']:.4f}")
            print(f"    λ: {priors['lambda']['mean']:.4f}"
                  f"(up: {priors['lambda_up']['mean']:.4f}, "
                  f"down: {priors['lambda_down']['mean']:.4f})")
            print(f"    μ_J: {priors['mu_J']['mean']:+.4f}"
                   f"(up: {priors['mu_J_up']['mean']:+.4f}, "
                  f"down: {priors['mu_J_down']['mean']:+.4f})")
            print(f"    σ_J: {priors['sigma_J']['mean']:.4f}"
                  f"(up: {priors['sigma_J_up']['mean']:.4f}, "
                  f"down: {priors['sigma_J_down']['mean']:.4f})")
            print(f"    n_players: {priors['n_players']}")
    
    if age_groups:
        print(f"\n AGE-GROUP PRIORS (most specific):")
        print("-" * 50)
        for (position, age_group), priors in age_groups.items():
            print(f"\n  {position}, {age_group}:")
            print(f"    μ: {priors['mu']['mean']:.4f} ± {priors['mu']['std']:.4f}")
            print(f"    σ: {priors['sigma']['mean']:.4f} ± {priors['sigma']['std']:.4f}")
            print(f"    λ: {priors['lambda']['mean']:.4f}"
                  f"(up: {priors['lambda_up']['mean']:.4f}, "
                  f"down: {priors['lambda_down']['mean']:.4f})")
            print(f"    μ_J: {priors['mu_J']['mean']:+.4f}"
                  f"(up: {priors['mu_J_up']['mean']:+.4f}, "
                  f"down: {priors['mu_J_down']['mean']:+.4f})")
            print(f"    σ_J: {priors['sigma_J']['mean']:.4f}"
                  f"(up: {priors['sigma_J_up']['mean']:.4f}, "
                  f"down: {priors['sigma_J_down']['mean']:.4f})")
            print(f"    n_players: {priors['n_players']}")


if __name__ == "__main__":
    prior_db = build_prior_database(min_valuations=10, verbose=True)
    
    if prior_db:
        save_prior_database(prior_db)
        print(f"\n Prior database built with {len(prior_db)} groups")
        inspect_prior_database()
    else:
        print("\n Failed to build prior database")