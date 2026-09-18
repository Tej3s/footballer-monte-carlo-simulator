#using monte carlo simulations to run jump-diffusion model for any footballer

from os import path

from matplotlib.pylab import seed
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import your Hard EM class

from hard_em import HardEMJumpDiffusion  # Your existing class

import pickle
from scipy.stats import genpareto
from gpd_tail import load_gpd, sample_excess

class FootballerMonteCarlo:
    def __init__(self,
                 player_name: str,
                  initial_value: float,
                 results: Dict,  # ← Direct from Hard EM's fit() method
                 log_returns: np.ndarray = None,
                 years: int = 2,
                 time_steps_per_year: int = 4,
                 n_simulations: int = 10000,
                 seed: Optional[int] = None):

        self.player_name = player_name
        self.initial_value = initial_value

        # Extract parameters from Hard EM results
        self.mu = results['mu']
        self.sigma = results['sigma']
        self.n_jumps_actual = results['n_jumps']
        self.n_observations = results['n_observations']

        # Asymmetric jump parameters (with fallbacks for old-format results)
        self.lambda_up = results.get('lambda_up', results['lambda'])
        self.lambda_down = results.get('lambda_down', 0.0)
        self.mu_J_up = results.get('mu_J_up', results['mu_J'])
        self.mu_J_down = results.get('mu_J_down', 0.0)
        self.sigma_J_up = results.get('sigma_J_up', results['sigma_J'])
        self.sigma_J_down = results.get('sigma_J_down', 0.0)

        # Aggregates (kept for logging and plotting)
        self.lambda_jump = self.lambda_up + self.lambda_down
        self.mu_J = results['mu_J']
        self.sigma_J = results['sigma_J']

        self.years = years
        self.time_steps_per_year = time_steps_per_year
        self.n_simulations = n_simulations
        self.n_steps = years * time_steps_per_year
        self.dt = 1 / time_steps_per_year  # Time step in years

        self.log_returns = np.asarray(log_returns) if log_returns is not None else None

        self.gpd_params = load_gpd()
        self.pct_extreme = self.compute_pct_extreme(results)


        if seed is not None:
            np.random.seed(seed)

        self.simulations = None
        self.statistics = None


        print(f"\n{'='*60}")
        print(f"Monte Carlo: {self.player_name}")
        print(f"{'='*60}")
        print(f"   Parameters from Hard EM:")
        print(f"      μ (drift): {self.mu:.4f}")
        print(f"      σ (volatility): {self.sigma:.4f}")
        print(f"      λ_up: {self.lambda_up:.4f}, λ_down: {self.lambda_down:.4f}")
        print(f"      μ_J_up: {self.mu_J_up:.4f}, μ_J_down: {self.mu_J_down:.4f}")
        print(f"      σ_J_up: {self.sigma_J_up:.4f}, σ_J_down: {self.sigma_J_down:.4f}")
        print(f"      Aggregate λ: {self.lambda_jump:.4f}, μ_J: {self.mu_J:.4f}, σ_J: {self.sigma_J:.4f}")
        print(f"      Actual jumps detected: {self.n_jumps_actual}")
        print(f"      Observations: {self.n_observations}")
        print(f"   Initial Value: €{self.initial_value:,.0f}")
        print(f"   Time Horizon: {self.years} years ({self.n_steps} time steps)")
        print(f"   Simulations: {self.n_simulations:,}")


    def compute_pct_extreme(self, results):

        jump_indices = results.get('jump_indices', [])
        if len(jump_indices) == 0 or self.log_returns is None:
            return 0.0
        threshold = self.gpd_params['threshold']
        jump_vals = np.array([self.log_returns[i] for i in jump_indices])
        return float(np.mean(jump_vals > threshold))

    def simulate_single_path(self):
        """Simulate a single path of the jump-diffusion process."""
        S = np.zeros(self.n_steps + 1)
        S[0] = self.initial_value

        for t in range(1, self.n_steps + 1):
            # Simulate the jump component
            current_value = S[t - 1]

            #GBM component
            epsilon = np.random.normal(0,1)
            dW = epsilon * np.sqrt(self.dt)

            gbm_component = current_value * np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * dW)

            # Jump component
            U = np.random.uniform(0, 1)

            lambda_total = self.lambda_up + self.lambda_down
            if lambda_total > 0 and U < lambda_total * self.dt:
                p_up = self.lambda_up / max(lambda_total, 1e-10)
                if np.random.uniform() < p_up:
                    log_jump = self.mu_J_up + self.sigma_J_up * np.random.normal()
                else:
                    log_jump = self.mu_J_down + self.sigma_J_down * np.random.normal()

         # Upward GPD cap (only applies to upward jumps)
                threshold = self.gpd_params['threshold']
                if log_jump >= threshold:
                    excess = genpareto.rvs(
                        self.gpd_params['shape'],
                        loc=0,
                        scale=self.gpd_params['scale']
                    )
                    log_jump = threshold + excess

                new_value = gbm_component * np.exp(log_jump)
            else:
                new_value = gbm_component

            S[t] = new_value

        return S

    def run_simulations(self, verbose: bool = True):
        """Run multiple simulations and store the results."""
        if verbose:
            print(f"\nRunning {self.n_simulations:,} simulations...")   

        all_paths = np.zeros((self.n_simulations, self.n_steps + 1))

        for i in range(self.n_simulations):
            path = self.simulate_single_path()
            all_paths[i] = path

            if verbose and (i + 1) % (self.n_simulations // 10) == 0:
                print(f"   Completed {i + 1:,}/{self.n_simulations:,} simulations")

        self.simulations = all_paths
        self.compute_statistics()

        if verbose:
            self._print_statistics()

        return self.statistics

    def compute_statistics(self):

        paths = self.simulations
        times = np.arange(self.n_steps + 1) * self.dt * 52

        final_values = paths[:, -1]

        self.statistics = {
             'times': times,
            'mean': np.mean(paths, axis=0),
            'median': np.median(paths, axis=0),
            'std': np.std(paths, axis=0),
            'p5': np.percentile(paths, 5, axis=0),
            'p25': np.percentile(paths, 25, axis=0),
            'p75': np.percentile(paths, 75, axis=0),
            'p95': np.percentile(paths, 95, axis=0),
            'final_values': final_values,
            'mean_final': np.mean(final_values),
            'median_final': np.median(final_values),
            'std_final': np.std(final_values),
            'p5_final': np.percentile(final_values, 5),
            'p95_final': np.percentile(final_values, 95),
            'prob_increase': np.mean(final_values > self.initial_value),
            'prob_double': np.mean(final_values > 2 * self.initial_value),
            'prob_halve': np.mean(final_values < 0.5 * self.initial_value),
        }

    def _print_statistics(self):
            stats = self.statistics
            print(f"\nSimulation Statistics:")
            print(f"   Mean final value: €{stats['mean_final']:,.0f}")
            print(f"   Median final value: €{stats['median_final']:,.0f}")
            print(f"   Std dev of final values: €{stats['std_final']:,.0f}")
            print(f"   5th percentile final value: €{stats['p5_final']:,.0f}")
            print(f"   95th percentile final value: €{stats['p95_final']:,.0f}")
            print(f"   Probability of increase: {stats['prob_increase']:.2%}")
            print(f"   Probability of doubling: {stats['prob_double']:.2%}")
            print(f"   Probability of halving: {stats['prob_halve']:.2%}")

    def plot_simulations(self, save_path: Optional[str] = None):

            if self.simulations is None:
                raise ValueError("Simulations have not been run yet. Call run_simulations() first.")
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            s = self.statistics
            times = s['times']
        
        # 1. Paths with confidence intervals
            ax = axes[0, 0]
            n_sample = min(100, self.n_simulations)
            sample_idx = np.random.choice(self.n_simulations, n_sample, replace=False)
            for idx in sample_idx:
                ax.plot(times, self.simulations[idx], alpha=0.05, color='blue')
            ax.fill_between(times, s['p5'], s['p95'], alpha=0.2, color='blue', label='5%-95%')
            ax.fill_between(times, s['p25'], s['p75'], alpha=0.3, color='blue', label='25%-75%')
            ax.plot(times, s['mean'], 'r-', label='Mean', linewidth=2)
            ax.axhline(y=self.initial_value, color='gray', linestyle=':', label='Initial')
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Market Value (€)')
            ax.set_title(f'{self.player_name}: Valuation Paths')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Final value distribution
            ax = axes[0, 1]
            ax.hist(s['final_values'], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=self.initial_value, color='red', linestyle='--', 
                  label=f'Initial: €{self.initial_value:,.0f}')
            ax.axvline(x=s['mean_final'], color='green', 
                  label=f'Mean: €{s["mean_final"]:,.0f}')
            ax.set_xlabel('Final Market Value (€)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{self.player_name}: Final Value Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Jump distribution
            ax = axes[1, 0]
        # Calculate jump counts per simulation (approximate)
            expected_jumps = self.lambda_jump * self.years
            jump_counts = np.random.poisson(expected_jumps, self.n_simulations)
            ax.hist(jump_counts, bins=range(0, int(max(jump_counts)) + 2), 
                alpha=0.7, color='green', edgecolor='black')
            ax.axvline(x=expected_jumps, color='red', linestyle='--',
                  label=f'Expected: {expected_jumps:.1f}')
            ax.set_xlabel('Number of Jumps')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{self.player_name}: Jump Distribution\n'
             f'(λ_up={self.lambda_up:.3f}, λ_down={self.lambda_down:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Probability of exceeding thresholds
            ax = axes[1, 1]
            thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            for thresh in thresholds:
                prob = np.mean(self.simulations > self.initial_value * thresh, axis=0)
                ax.plot(times, prob, label=f'>{thresh:.1f}x', linewidth=2)
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Probability')
            ax.set_title(f'{self.player_name}: Probability of Exceeding Thresholds')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
            plt.tight_layout()
        
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Figure saved to: {save_path}")
        
            plt.show()

    def get_value_at_risk(self, confidence_level: float = 0.95):
        """Calculate the Value at Risk (VaR) at a given confidence level."""
        if self.simulations is None:
            raise ValueError("Simulations have not been run yet. Call run_simulations() first.")
        
        final_values = self.simulations[:, -1]
        var = np.percentile(final_values, (1-confidence_level) * 100)
        cvar = np.mean(final_values[final_values <= var])  # Conditional VaR (Expected Shortfall)
        upside = np.percentile(final_values, (confidence_level) * 100)

        return{
            'VaR': var,
            'CVaR': cvar,
            'Upside': upside
        }

    