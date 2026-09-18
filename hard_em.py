

from tabnanny import verbose



import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

from scipy.stats import poisson, norm

import os
import sys

# Add the parent directory to Python's path so it can find files
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Change working directory to the parent where CSV files are
os.chdir(parent_dir)


# Connect to the database
conn = duckdb.connect('transfermarkt-datasets.duckdb')


players = pd.read_csv('players.csv')
valuations = pd.read_csv('player_valuations.csv')
appearances = pd.read_csv('appearances.csv')
transfers = pd.read_csv('transfers.csv')
events = pd.read_csv('game_events.csv')
comps = pd.read_csv('competitions.csv')
nt = pd.read_csv('national_teams.csv')
countries = pd.read_csv('countries.csv')
games = pd.read_csv('games.csv')
club_games = pd.read_csv('club_games.csv')

clubs = pd.read_csv('clubs.csv')


valuations['date'] = pd.to_datetime(valuations['date'])

def get_player_log_returns(player_id = None, min_valuations=2, end_date= None):
    vals = valuations.copy()
    vals = vals.sort_values(['player_id', 'date'])

    if end_date is not None:
        cutoff = pd.Timestamp(end_date)
        vals = vals[vals['date'] <= cutoff]


    all_results = []

    if player_id is not None:
        player_ids = [player_id]
    else:
        player_ids = vals['player_id'].unique()

    
    for pid in player_ids:
        group = vals[vals['player_id'] == pid]
        player_info = players[players['player_id'] == pid]
        
        if len(player_info) == 0:
            continue
        
        values = group['market_value_in_eur'].values
        dates = group['date'].values
        
        if len(values) < min_valuations:
            continue
  

        log_returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0 and values[i] > 0:
                log_returns.append(np.log(values[i] / values[i-1]))
        
        if len(log_returns) < 3:
            continue

               # === COMPUTE AGE FROM date_of_birth ===
        age_at_valuation = None
        dob_col = None
        for candidate in ['date_of_birth']:
            if candidate in player_info.columns:
                dob_col = candidate
                break
        
        if dob_col is not None:
            dob_str = player_info.iloc[0].get(dob_col)
            if pd.notna(dob_str):
                try:
                    dob = pd.to_datetime(dob_str)
                    last_val_date = pd.to_datetime(dates[-1])
                    age_at_valuation = (last_val_date - dob).days / 365.25
                except Exception:
                    age_at_valuation = None


        all_results.append({
            'player_id': pid,
            'name': player_info.iloc[0]['name'],
            'position': player_info.iloc[0]['position'],
            'age': age_at_valuation,
            'mu': np.mean(log_returns),
            'sigma': np.std(log_returns),
            'n_valuations': len(values),
            'n_log_returns': len(log_returns),
            'dates': dates.tolist(),
            'first_date': dates[0],
            'last_date': dates[-1],
            'current_value': values[-1],
            'peak_value': np.max(values),
            'log_returns': log_returns
        })
    
    return pd.DataFrame(all_results)

pedri_only = get_player_log_returns(player_id=683840)
print(pedri_only[['name', 'mu', 'sigma', 'n_log_returns']])


all_players = get_player_log_returns(min_valuations = 2)
all_players.to_csv('all_players_log_returns.csv', index=False)
print("💾 Saved all_players to CSV")



##implementing hard-em
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')



class HardEMJumpDiffusion:

    def __init__(self, max_iter: int = 100, tol: float = 1e-6, kmax: int = 2):
        self.max_iter = max_iter
        self.tol = tol
        self.history = []
        self.kmax = kmax 

    
    @staticmethod
    def find_optimal_window_em(log_returns, window_range=(3, 20) ):

        n = len(log_returns)

        if n < 10:
            return max(2, n//2) 
        
        best_window = 10
        best_stability = -np.inf

        for window in range(window_range[0], min(window_range[1], n//2)+1):
            #computing local volatility
            vols = []
            for i in range(window, n):
                recent = log_returns[i-window:i]
                vols.append(np.std(recent))

            cv = np.std(vols) / (np.mean(vols) + 1e-10)
            stability = 1/(cv + 1e-10)

            size_penalty = np.exp(-window/20)

            score = stability*(1+0.5 * size_penalty)

            if score > best_stability:
                best_stability = score
                best_window = window

        return best_window

    @staticmethod
    def find_optimal_c_percentile(log_returns, window, target_percentile=0.97):
    
        n = len(log_returns)
        ratios = []
    
        for i in range(window, n):
            # Only consider non-zero returns
            if abs(log_returns[i]) < 1e-10:
                continue
        
            recent = log_returns[i-window:i]
            local_vol = np.std(recent)
            if local_vol > 0:
                ratios.append(abs(log_returns[i]) / local_vol)
    
        if len(ratios) < 5:
            return 2.5  # Not enough data, use default from Lee-Mykland
    
        C = np.percentile(ratios, target_percentile * 100)
        C = max(2.0, min(4.5, C))  # Stricter lower bound
    
        return C

# hard_em.py - Add this at the bottom

    

    def fit(self, log_returns: np.ndarray, dt: float = 1.0, verbose: bool = True,
            
        )-> Dict:
        n = len(log_returns)
        x = np.array(log_returns)

        optimal_window = self.find_optimal_window_em(log_returns)
        optimal_C = self.find_optimal_c_percentile(log_returns, window=optimal_window, target_percentile = 0.95)
        
        initial_jumps = []
        for i in range(optimal_window, n):
            recent = log_returns[i-optimal_window:i]
            local_vol = np.std(recent)
            if local_vol > 0:
                threshold = optimal_C * local_vol
                if abs(log_returns[i]) > threshold:
                    initial_jumps.append(i)
                                                
                        # After EM assigns jumps, verify they meet the threshold
# If not, force them to be non-jumps

        if verbose:
            print(f"Initial jumps: {len(initial_jumps)}")
            print(f"Optimal window size: {optimal_window}")

             
        mu = np.mean(x)
        sigma = np.std(x)

        lamb = 0.10  # Initial guess: 10% jump probability
        lamb_up = 0.05
        lamb_down = 0.05
        mu_J_up = 0.0
        mu_J_down = 0.0   # Mean jump size (log-normal)
        sigma_J_up = 0.3  # Jump volatility
        sigma_J_down = 0.3
        

        self.history = []

        if len(initial_jumps) > 0:
            jump_returns = [x[i] for i in initial_jumps]

            up_jumps =  [r for r in jump_returns if r > 0]
            down_jumps =  [r for r in jump_returns if r < 0]

            lamb_up = max(len(up_jumps) / n, 0.0)
            lamb_down = max(len(down_jumps) / n, 0.0)

            if up_jumps:
                mu_J_up = np.mean(up_jumps)
                sigma_J_up =  max(np.std(up_jumps), 0.05)

            else: 
                mu_J_up = 0.0
                sigma_J_up = 0.0

            if down_jumps:
                mu_J_down = np.mean(down_jumps)
                sigma_J_down = max(np.std(down_jumps), 0.05)

            else:
                mu_J_down = 0.0
                sigma_J_down = 0.0

        lamb = lamb_up + lamb_down
        if lamb > 0:
            mu_J = (mu_J_up * lamb_up + mu_J_down * lamb_down) / lamb
            sigma_J2 = (sigma_J_up**2 * lamb_up + sigma_J_down**2 * lamb_down) / lamb
            sigma_J = np.sqrt(max(sigma_J2, 1e-10))
        else:
            mu_J = 0.0
            sigma_J = 0.0

        self.history = []



    
        if verbose:
            print(f"\n   Initial parameters:")
            print(f"   μ = {mu:.4f}, σ = {sigma:.4f}")
            print(f"   λ_up = {lamb_up:.4f}, λ_down = {lamb_down:.4f}")
            print(f"   μ_J_up = {mu_J_up:.4f}, μ_J_down = {mu_J_down:.4f}")
            print(f"   σ_J_up = {sigma_J_up:.4f}, σ_J_down = {sigma_J_down:.4f}")

       
        k_t = np.zeros(n, dtype=int)
        jump_indices = np.array([], dtype=int)
        n_jumps = 0
        iteration = 0


        
        for iteration in range(self.max_iter):
            k_t = np.zeros(n, dtype=int)

            for i in range(n):
                likelihoods = np.zeros(self.kmax+1)

                for k in range(self.kmax+1):
                     
                    if x[i] >= 0:
                        mu_k = (mu ) + k * mu_J_up
                        sigma_k2 = sigma**2 + k * sigma_J_up**2

                        prior_k = poisson.pmf(k, lamb_up * dt) if lamb_up > 0 else (1.0 if k == 0 else 0.0)

                       

                    else:
                        mu_k = mu + k*mu_J_down
                        sigma_k2 = sigma**2 + k * sigma_J_down**2
                        prior_k = poisson.pmf(k, lamb_down * dt) if lamb_down > 0 else (1.0 if k == 0 else 0.0)
                        
                    sigma_k = np.sqrt(max(sigma_k2, 1e-10))
                    likelihoods[k] = prior_k * norm.pdf(x[i], mu_k, sigma_k)
                   
               
                k_t[i] = np.argmax(likelihoods)

                   
                

            jump_indices = np.where(k_t > 0)[0]
            n_jumps = len(jump_indices)

            if n_jumps == 0:

                mu_new = np.mean(x)
                sigma_new = np.std(x)
                lamb_up_new = lamb_up * 0.95
                lamb_down_new = lamb_down * 0.95
                mu_J_up_new = mu_J_up
                mu_J_down_new = mu_J_down
                sigma_J_up_new = sigma_J_up
                sigma_J_down_new = sigma_J_down
            else:
                up_idx = [i for i in jump_indices if x[i] > 0]
                down_idx = [i for i in jump_indices if x[i] < 0]

                lamb_up_new = len(up_idx) / (n * dt)
                lamb_down_new = len(down_idx) / (n * dt)

            
                # Upward jumps
                if len(up_idx) >= 3:
                    mu_J_up_new = np.mean(x[up_idx]) - mu
                    sigma_J_up_new = np.std(x[up_idx] - mu - mu_J_up_new)
                elif len(up_idx) >= 1:
                    mu_J_up_new = np.mean(x[up_idx]) - mu
                    sigma_J_up_new = max(sigma * 0.5, 0.10)
                else:
                    mu_J_up_new = 0.0
                    sigma_J_up_new = 0.0


                 # Downward jumps
                if len(down_idx) >= 3:
                    mu_J_down_new = np.mean(x[down_idx]) - mu
                    sigma_J_down_new = np.std(x[down_idx] - mu - mu_J_down_new)
                elif len(down_idx) >= 1:
                    mu_J_down_new = np.mean(x[down_idx]) - mu
                    sigma_J_down_new = max(sigma * 0.5, 0.10)
                else:
                    mu_J_down_new = 0.0
                    sigma_J_down_new = 0.0

                k_up = np.array([1 if (k_t[i] > 0 and x[i] >= 0) else 0 for i in range(n)])
                k_down = np.array([1 if (k_t[i] > 0 and x[i] < 0) else 0 for i in range(n)])

                mu_new = np.sum(x - k_up * mu_J_up_new - k_down * mu_J_down_new) / n

                resid = x - mu_new - k_up * mu_J_up_new - k_down * mu_J_down_new
                jump_var = k_up * sigma_J_up_new**2 + k_down * sigma_J_down_new**2
                sigma2_new = np.sum(resid**2 - jump_var) / n
                sigma_new = np.sqrt(max(sigma2_new, 1e-10))

        # === Compute aggregates from *_new values (weighted variance, not quadrature) ===
            lamb_new = lamb_up_new + lamb_down_new
            if lamb_new > 0:
                mu_J_new = (mu_J_up_new * lamb_up_new + mu_J_down_new * lamb_down_new) / lamb_new
                sigma_J2_new = (sigma_J_up_new**2 * lamb_up_new + sigma_J_down_new**2 * lamb_down_new) / lamb_new
                sigma_J_new = np.sqrt(max(sigma_J2_new, 1e-10))
            else:
                mu_J_new = 0.0
                sigma_J_new = 0.0

          

          
            max_diff = max([
                abs(mu_new - mu),
                abs(sigma_new - sigma),
                abs(lamb_new - lamb),
                abs(mu_J_new - mu_J),
                abs(sigma_J_new - sigma_J)
            ])
            
            # Store history
            self.history.append({
                'iteration': iteration + 1,
                'mu': mu_new, 'sigma': sigma_new,
                'lambda_up': lamb_up_new, 'lambda_down': lamb_down_new,
                'mu_J_up': mu_J_up_new, 'mu_J_down': mu_J_down_new,
                'sigma_J_up': sigma_J_up_new, 'sigma_J_down': sigma_J_down_new,
                'lambda': lamb_new, 'mu_J': mu_J_new, 'sigma_J': sigma_J_new,
                'n_jumps': n_jumps, 'max_diff': max_diff,
            })
            
            # Update parameters
            mu, sigma = mu_new, sigma_new
            lamb_up, lamb_down = lamb_up_new, lamb_down_new
            mu_J_up, mu_J_down = mu_J_up_new, mu_J_down_new
            sigma_J_up, sigma_J_down = sigma_J_up_new, sigma_J_down_new
            lamb, mu_J, sigma_J = lamb_new, mu_J_new, sigma_J_new
            
            if verbose and iteration % 10 == 0:
                print(f"\n   Iteration {iteration+1}:")
                print(f"      μ = {mu:.4f}, σ = {sigma:.4f}")
                print(f"      λ_up = {lamb_up:.4f}, λ_down = {lamb_down:.4f}")
                print(f"      μ_J_up = {mu_J_up:.4f}, μ_J_down = {mu_J_down:.4f}")
                print(f"      jumps = {n_jumps}, max_diff = {max_diff:.6f}")

            
            if max_diff < self.tol:
                if verbose:
                    print(f"Converged after {iteration+1} iterations!")
                break


        if n_jumps > 0:
            n_up = len([i for i in jump_indices if x[i] > 0])
            n_down = len([i for i in jump_indices if x[i] < 0])
        else:
            n_up, n_down = 0, 0
        
       
        # Final parameters
        results = {
            'mu': mu, 'sigma': sigma,
        # Aggregates for RLS and MC
            'lambda': lamb, 'mu_J': mu_J, 'sigma_J': sigma_J,
        # Asymmetric split
            'lambda_up': lamb_up, 'lambda_down': lamb_down,
            'mu_J_up': mu_J_up, 'mu_J_down': mu_J_down,
            'sigma_J_up': sigma_J_up, 'sigma_J_down': sigma_J_down,
            'n_jumps_up': n_up, 'n_jumps_down': n_down,
            'n_jumps': n_jumps,
            'iterations': iteration + 1,
            'n_observations': n,
            'history': self.history,
            'k_t': k_t,
            'jump_indices': jump_indices if n_jumps > 0 else []
        }
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        print(f"drift: {results['mu']:.4f}")
        print(f"   σ (volatility):     {results['sigma']:.4f}")
        print(f"   λ (jump freq):      {results['lambda']:.4f}")
        print(f"   μ_J (jump mean):    {results['mu_J']:.4f}")
        print(f"   σ_J (jump std):     {results['sigma_J']:.4f}")
        print(f"   Jumps detected:     {results['n_jumps']}")
        print(f"   Iterations:         {results['iterations']}")
        print(f"   Observations:       {results['n_observations']}")


    @staticmethod
    def analyze_hard_em_results(results: Dict, log_returns: List[float]):
        k_t = results['k_t']
       
        n_jumps = results['n_jumps']
        mu_J = results['mu_J']
        sigma_J = results['sigma_J']
        
        
        if n_jumps > 0:
            jump_indices = results['jump_indices']
            jump_returns = [log_returns[i] for i in jump_indices]
            print(f"\n2. Jump Sizes:")
            print(f"   Mean jump log-return: {np.mean(jump_returns):.4f}")
            print(f"   Std jump log-return: {np.std(jump_returns):.4f}")
            print(f"   Largest jump: {np.max(jump_returns):.4f}")
            print(f"   Smallest jump: {np.min(jump_returns):.4f}")
        else:
            print("No jumps detected in the data.")    

        history = results['history']
        if history:
            print(f"\n3. Convergence:")
            print(f"   Iterations: {results['iterations']}")
            print(f"   Final max_diff: {history[-1]['max_diff']:.6f}")
    

# In a new session or different script
pedri_log_returns = pedri_only.iloc[0]['log_returns']
pedri_name = pedri_only.iloc[0]['name']


print("=" * 60)



# Run Hard EM
hard_em = HardEMJumpDiffusion(max_iter=50, kmax=1)
results = hard_em.fit(pedri_log_returns, dt=1.0)
optimal_window = hard_em.find_optimal_window_em(pedri_log_returns)
print(f"\nOptimal window size for local volatility estimation: {optimal_window}")
# Print results
hard_em._print_summary(results)

best_C = hard_em.find_optimal_c_percentile(pedri_log_returns, window=optimal_window, target_percentile=0.95)
print(f"Optimal C: {best_C:.2f}")

# Extract parameters
mu = results['mu']
sigma = results['sigma']
lambda_jump = results['lambda']
mu_J = results['mu_J']
sigma_J = results['sigma_J']

print(f"\n Pedri's Jump-Diffusion Parameters:")
print(f"   μ = {mu:.4f}")
print(f"   σ = {sigma:.4f}")
print(f"   λ = {lambda_jump:.4f}")
print(f"   μ_J = {mu_J:.4f}")
print(f"   σ_J = {sigma_J:.4f}")



lewa_players = players[players['name'].str.contains('Lewandowski', case=False, na=False)]


# Get the correct ID (use the one from your data)
# Based on typical Transfermarkt data, it might be 132847, but let's find it


# Get log returns for Lewandowski
lewa_data = get_player_log_returns(player_id=  38253)
print(f"\n📊 Lewandowski's data:")
print(lewa_data[['name', 'mu', 'sigma', 'n_valuations', 'n_log_returns']])

# Extract log returns
lewa_log_returns = lewa_data.iloc[0]['log_returns']
lewa_name = lewa_data.iloc[0]['name']

print(f"\n Lewandowski's Log Returns:")
print(lewa_log_returns)
print(f"\n Summary:")
print(f"   Number of log returns: {len(lewa_log_returns)}")
print(f"   Mean return: {np.mean(lewa_log_returns):.4f}")
print(f"   Std return: {np.std(lewa_log_returns):.4f}")

# Run Hard EM on Lewandowski
print(f"\n{'='*60}")
print(f"Running Hard EM on {lewa_name}")
print("="*60)

hard_em = HardEMJumpDiffusion(max_iter=50, kmax=1)
results_lewa = hard_em.fit(lewa_log_returns, dt=1.0)

# Get optimal parameters for diagnostics
optimal_window = hard_em.find_optimal_window_em(lewa_log_returns)
optimal_C = hard_em.find_optimal_c_percentile(lewa_log_returns, window=optimal_window)
print(f"\nOptimal window size: {optimal_window}")
print(f"Optimal C: {optimal_C:.2f}")

# Extract parameters
mu = results_lewa['mu']
sigma = results_lewa['sigma']
lambda_jump = results_lewa['lambda']
mu_J = results_lewa['mu_J']
sigma_J = results_lewa['sigma_J']

print(f"\n {lewa_name}'s Jump-Diffusion Parameters:")
print(f"   μ = {mu:.4f}")
print(f"   σ = {sigma:.4f}")
print(f"   λ = {lambda_jump:.4f}")
print(f"   μ_J = {mu_J:.4f}")
print(f"   σ_J = {sigma_J:.4f}")
print(f"   Jumps detected: {results_lewa['n_jumps']}")
print(f"   Jump indices: {results_lewa['jump_indices']}")

# Diagnostic check - see how many jumps meet the C threshold
if results_lewa['n_jumps'] > 0:
    valid_count = 0
    for idx in results_lewa['jump_indices']:
        if idx >= optimal_window:
            recent = lewa_log_returns[idx-optimal_window:idx]
            local_vol = np.std(recent)
            if local_vol > 0:
                threshold = optimal_C * local_vol
                if abs(lewa_log_returns[idx]) > threshold:
                    valid_count += 1
    
    print(f"\n Jump Quality Check (diagnostic only):")
    print(f"   {valid_count}/{results_lewa['n_jumps']} jumps meet the C={optimal_C:.2f} threshold")

print(" Finding Lewandowski's player ID...")
lewa_players = players[players['name'].str.contains('Lewandowski', case=False, na=False)]
print(lewa_players[['player_id', 'name', 'position', 'last_season']])

# Get the correct ID (use the one from your data)
# Based on typical Transfermarkt data, it might be 132847, but let's find it
lewa_id = lewa_players.iloc[0]['player_id']  # Use the first match
lewa_name = lewa_players.iloc[0]['name']

print(f"\n Found: {lewa_name} (ID: {lewa_id})")

# Get log returns for Lewandowski
lewa_data = get_player_log_returns(player_id=lewa_id)

print(lewa_data[['name', 'mu', 'sigma', 'n_valuations', 'n_log_returns']])

# Extract log returns
lewa_log_returns = lewa_data.iloc[0]['log_returns']
lewa_name = lewa_data.iloc[0]['name']

print(lewa_log_returns)
print(f"\n Summary:")
print(f"   Number of log returns: {len(lewa_log_returns)}")
print(f"   Mean return: {np.mean(lewa_log_returns):.4f}")
print(f"   Std return: {np.std(lewa_log_returns):.4f}")

# Run Hard EM on Lewandowski
print(f"\n{'='*60}")
print(f" Running Hard EM on {lewa_name}")
print("="*60)

hard_em = HardEMJumpDiffusion(max_iter=50, kmax=1)
results_lewa = hard_em.fit(lewa_log_returns, dt=1.0)

# Get optimal parameters for diagnostics
optimal_window = hard_em.find_optimal_window_em(lewa_log_returns)
optimal_C = hard_em.find_optimal_c_percentile(lewa_log_returns, window=optimal_window)
print(f"\nOptimal window size: {optimal_window}")
print(f"Optimal C: {optimal_C:.2f}")

# Extract parameters
mu = results_lewa['mu']
sigma = results_lewa['sigma']
lambda_jump = results_lewa['lambda']
mu_J = results_lewa['mu_J']
sigma_J = results_lewa['sigma_J']

print(f"\n {lewa_name}'s Jump-Diffusion Parameters:")
print(f"   μ = {mu:.4f}")
print(f"   σ = {sigma:.4f}")
print(f"   λ = {lambda_jump:.4f}")
print(f"   μ_J = {mu_J:.4f}")
print(f"   σ_J = {sigma_J:.4f}")
print(f"   Jumps detected: {results_lewa['n_jumps']}")
print(f"   Jump indices: {results_lewa['jump_indices']}")

# Diagnostic check - see how many jumps meet the C threshold
if results_lewa['n_jumps'] > 0:
    valid_count = 0
    for idx in results_lewa['jump_indices']:
        if idx >= optimal_window:
            recent = lewa_log_returns[idx-optimal_window:idx]
            local_vol = np.std(recent)
            if local_vol > 0:
                threshold = optimal_C * local_vol
                if abs(lewa_log_returns[idx]) > threshold:
                    valid_count += 1
    
    print(f"\n Jump Quality Check (diagnostic only):")
    print(f"   {valid_count}/{results_lewa['n_jumps']} jumps meet the C={optimal_C:.2f} threshold")
    
   