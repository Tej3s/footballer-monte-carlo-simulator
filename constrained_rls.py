

import numpy as np
from typing import Dict, Optional, Tuple
import pickle
import os


from finding_priors import load_prior_database


class ConstrainedRLS:
    """
    RLS for edge cases. All edge cases follow the same pattern:
    - Extreme parameters
    - Limited data
    - Few jumps
    - σ_J = 0
    
    Therefore, they use the SAME regularization weights.
    """
    
    def __init__(self, prior_db_path: str = 'prior_db.pkl'):
        """Initialize with prior database."""
        # ✅ Load from file
        self.prior_db = load_prior_database(prior_db_path)
        
        # === EDGE CASE WEIGHTS (SAME FOR ALL) ===
        self.edge_case_weights = {
            'extreme_data_weight': 0.25,
            'extreme_prior_weight': 0.75,
            'normal_data_weight': 0.55,
            'normal_prior_weight': 0.45,
            'sigma_j_prior': 0.12,
        }
        
        # ✅ Extract from database
        self.position_priors = self._extract_position_priors()
        self.age_priors = self.extract_age_priors()
        
        # Default priors (fallback)
        self.default_priors = {
            'mu': 0.05, 'sigma': 0.35, 'lambda': 0.015,
            'mu_J': 0.0, 'sigma_J': 0.15,
        }
        
        # Edge case thresholds
        self.thresholds = {
            'sigma_j_min': 0.001,
            'mu_max': 0.20,
            'mu_min': -0.15,
            'sigma_max': 0.55,
            'sigma_min': 0.05,
            'mu_j_max': 0.50,
            'n_obs_min': 8,
            'n_jumps_min': 2,
            'wonderkid_age': 22,
            'wonderkid_n_obs': 12,
        }
        
        
        print(f"   Edge case weights: data={self.edge_case_weights['extreme_data_weight']:.2f}, prior={self.edge_case_weights['extreme_prior_weight']:.2f}")
    
    def _extract_position_priors(self) -> Dict:
        """Extract position priors from the loaded database."""
        position_priors = {}
        
        if self.prior_db:
            for position in ['Attack', 'Midfield', 'Defender', 'Goalkeeper']:
                if position in self.prior_db:
                    p = self.prior_db[position]
                    position_priors[position] = {
                        'mu': p['mu']['mean'],
                        'sigma': p['sigma']['mean'],
                        'mu_J': p['mu_J']['mean'],
                        'sigma_J': p['sigma_J']['mean'], 'lambda': p['lambda']['mean'],   
                    }
                elif (position, 'all') in self.prior_db:
                    p = self.prior_db[(position, 'all')]
                    position_priors[position] = {
                        'mu': p['mu']['mean'],
                        'sigma': p['sigma']['mean'],
                        'mu_J': p['mu_J']['mean'],
                        'sigma_J': p['sigma_J']['mean'], 'lambda': p['lambda']['mean'],   
                    }
        
        # Fallback to defaults if database is empty
        if not position_priors:
            print("  No priors in database, using defaults")
            for position in ['Attack', 'Midfield', 'Defender', 'Goalkeeper']:
                position_priors[position] = {
                    'mu': 0.05,
                    'sigma': 0.18,
                    'mu_J': 0.15,
                    'sigma_J': 0.10,
                    'lambda': 0.015,
                }
        
        return position_priors

    def extract_age_priors(self) -> Dict:

        age_priors = {}

        if self.prior_db:
            for key, p in self.prior_db.items():
                if isinstance(key, tuple):
                    pos, age_group = key
                    age_priors[(pos, age_group)] = {
                        'mu': p['mu']['mean'],
                        'sigma': p['sigma']['mean'],
                        'mu_J': p['mu_J']['mean'],
                        'sigma_J': p['sigma_J']['mean'],
                        'lambda': p['lambda']['mean'],
                        'mu_J_up': p['mu_J_up']['mean'],
                        'mu_J_down': p['mu_J_down']['mean'],
                        'sigma_J_up': p['sigma_J_up']['mean'],
                        'sigma_J_down': p['sigma_J_down']['mean'],
                        'lambda_up': p['lambda_up']['mean'],
                        'lambda_down': p['lambda_down']['mean'],
                    }
        return age_priors
    
    def age_to_group(self, age: Optional[int]) -> Optional[str]:
    """Convert numeric age to age group label."""
        if age is None:
            return None
        if age < 22:
            return 'young'
        elif age < 24:
            return 'prime_early'
        elif age < 26:
            return 'prime_mid'
        elif age < 28:
            return 'prime_later'
        elif age < 30:
            return 'prime_even_later'
        elif age < 32:
            return 'veteran'
        elif age < 34:
            return 'aging_veteran'
        else:
            return 'aged'

    def get_priors(self, position: str, age: Optional[int]=None) -> Dict:
        """Get priors for a specific position."""

        age_group = self.age_to_group(age)
        
        # Try position-specific priors first
        if age_group and (position, age_group) in self.age_priors:
            p = self.age_priors[(position, age_group)]
            return {
                'mu_0': p['mu'],
                'sigma_0': p['sigma'],
                'mu_j_0': p['mu_J'],
                'sigma_j_0': p['sigma_J'],
                'lambda_0': p['lambda'],
                 'mu_j_up_0': p['mu_J_up'],
                'mu_j_down_0': p['mu_J_down'],
                'sigma_j_up_0': p['sigma_J_up'],
                'sigma_j_down_0': p['sigma_J_down'],
                'lambda_up_0': p['lambda_up'],
                'lambda_down_0': p['lambda_down'],
                'position': f"{position},{age_group}",
            }
        
        # Try from database
        if position in self.position_priors:
            p = self.position_priors[position]
            return {
                 'mu_0': p['mu'],
                'sigma_0': p['sigma'],
                'mu_j_0': p['mu_J'],
                'sigma_j_0': p['sigma_J'],
                'lambda_0': p['lambda'],
                'position': position,
            }
        
        # Fallback to defaults
        return {
            'mu_0': self.default_priors['mu'],
            'sigma_0': self.default_priors['sigma'],
            'mu_j_0': self.default_priors['mu_J'],
            'sigma_j_0': self.default_priors['sigma_J'],
            'lambda_0': self.default_priors['lambda'],
            'position': 'default'
        }
    
    def is_edge_case(self, results: Dict, n_obs: int, age: Optional[int] = None) -> Tuple[bool, list]:
        """Detect edge cases."""
        
        reasons = []
        
        if results.get('sigma_J', 0) < self.thresholds['sigma_j_min']:
            reasons.append('sigma_J_zero')
        
        if results.get('sigma', 0) < 0.001:
            reasons.append('sigma_zero')
        
        mu = results.get('mu', 0)
        if mu > self.thresholds['mu_max']:
            reasons.append(f'mu_high_{mu:.4f}')
        if mu < self.thresholds['mu_min']:
            reasons.append(f'mu_low_{mu:.4f}')
        
        sigma = results.get('sigma', 0)
        if sigma > self.thresholds['sigma_max']:
            reasons.append(f'sigma_high_{sigma:.4f}')
        if sigma < self.thresholds['sigma_min']:
            reasons.append(f'sigma_low_{sigma:.4f}')
        
        mu_j = results.get('mu_J', 0)
        if abs(mu_j) > self.thresholds['mu_j_max']:
            reasons.append(f'mu_j_high_{mu_j:.4f}')
        
        if n_obs < self.thresholds['n_obs_min']:
            reasons.append(f'limited_data_{n_obs}')
        
        if results.get('n_jumps', 0) < self.thresholds['n_jumps_min']:
            reasons.append(f'few_jumps_{results.get("n_jumps", 0)}')
        
        if age is not None and age < self.thresholds['wonderkid_age']:
            if n_obs < self.thresholds['wonderkid_n_obs']:
                reasons.append(f'wonderkid_{age}_{n_obs}')
        
        return len(reasons) > 0, reasons
    
    def apply_rls(self, 
                  results: Dict, 
                  n_obs: int, 
                  position: str,
                  age: Optional[int] = None,
                  current_value: Optional[float] = None,
                  verbose: bool = False) -> Dict:
        """Apply RLS to edge cases."""
        
        priors = self.get_priors(position, age)
        fixed = results.copy()
        
        if n_obs <= 3:
            data_weight = 0.25
        elif n_obs >= 25:
            data_weight = 0.90
        else:
        # Linear interpolation between n=3 and n=25
            data_weight = 0.25 + 0.65 * (n_obs - 3) / 22
    
        prior_weight = 1.0 - data_weight
    
        # For symmetry with existing code, keep the variable names
        extreme_data_weight = data_weight
        extreme_prior_weight = prior_weight
        normal_data_weight = data_weight
        normal_prior_weight = prior_weight
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔧 RLS APPLIED (edge case)")
            print(f"{'='*60}")
            print(f"  Position: {position}")
            print(f"  Extreme data weight: {extreme_data_weight:.2f}")
            print(f"  Extreme prior weight: {extreme_prior_weight:.2f}")
        
        mu_data = results.get('mu', 0)
        sigma_data = results.get('sigma', 0.15)
        mu_j_data = results.get('mu_J', 0)
        sigma_j_data = results.get('sigma_J', 0)
        n_jumps = results.get('n_jumps', 0)
        
        mu_is_extreme = abs(mu_data) > self.thresholds['mu_max']
        mu_j_is_extreme = abs(mu_j_data) > self.thresholds['mu_j_max']
        sigma_is_extreme = sigma_data > self.thresholds['sigma_max']
        
        # === RLS: μ ===
        if mu_is_extreme:
            mu_new = extreme_data_weight * mu_data + extreme_prior_weight * priors['mu_0']
            if verbose:
                print(f"  μ (extreme): {mu_data:.4f} → {mu_new:.4f}")
        else:
            mu_new = normal_data_weight * mu_data + normal_prior_weight * priors['mu_0']
        
        if current_value:
            peak_value = 200_000_000
            proximity = min(current_value / peak_value, 1.0)
            growth_penalty = 1.0 - (proximity * 0.7)
            mu_new = mu_new * growth_penalty
        
        mu_new = np.clip(mu_new, -0.35, 0.27)
        fixed['mu'] = mu_new
        
        # === RLS: σ ===
        if sigma_is_extreme:
            sigma_new = extreme_data_weight * sigma_data + extreme_prior_weight * priors['sigma_0']
            if verbose:
                print(f"  σ (extreme): {sigma_data:.4f} → {sigma_new:.4f}")
        else:
            sigma_new = normal_data_weight * sigma_data + normal_prior_weight * priors['sigma_0']
        
        sigma_new = np.clip(sigma_new, 0.05, 0.35)
        fixed['sigma'] = sigma_new
        
        # === RLS: μ_J ===
        if mu_j_is_extreme or n_jumps < 2:
            mu_j_new = extreme_data_weight * mu_j_data + extreme_prior_weight * priors['mu_j_0']
            if verbose:
                print(f"  μ_J (extreme/unreliable): {mu_j_data:.4f} → {mu_j_new:.4f}")
        else:
            mu_j_new = normal_data_weight * mu_j_data + normal_prior_weight * priors['mu_j_0']
        
        #mu_j_new = np.clip(mu_j_new, -0.50, 0.50)
        fixed['mu_J'] = mu_j_new
        
        # === RLS: σ_J ===
        if n_jumps < 3:
            if n_jumps <= 1:
                sigma_j_new = priors['sigma_j_0']
                if verbose:
                    print(f"  σ_J (n_jumps={n_jumps}): {sigma_j_data:.4f} → {sigma_j_new:.4f} (prior)")
            else:
                sigma_j_new = np.sqrt(0.5 * sigma_j_data**2 + 0.5 * priors['sigma_j_0']**2)
        else:
            sigma_j_new = normal_data_weight * sigma_j_data + normal_prior_weight * priors['sigma_j_0']
        
        sigma_j_new = np.clip(sigma_j_new, 0.02, 0.40)
        fixed['sigma_J'] = sigma_j_new
        
        # === RLS: λ ===
        lambda_data = results.get('lambda', 0.08)
        lambda_prior = priors.get('lambda_0', 0.08)
        lambda_new = normal_data_weight * lambda_data + normal_prior_weight * lambda_prior
        lambda_new = np.clip(lambda_new, 0.005, 0.10)
        fixed['lambda'] = lambda_new

                # === Propagate shrinkage to asymmetric split parameters ===
        if 'mu_J_up' in results and 'mu_J_down' in results:
            lambda_up_raw = results.get('lambda_up', 0)
            lambda_down_raw = results.get('lambda_down', 0)
            lambda_raw = lambda_up_raw + lambda_down_raw
            
            if lambda_raw > 0:
                p_up = lambda_up_raw / lambda_raw
                p_down = lambda_down_raw / lambda_raw
                
                # Scale lambda splits to match shrunk aggregate
                fixed['lambda_up'] = lambda_new * p_up
                fixed['lambda_down'] = lambda_new * p_down
                
                # Rescale mu_J splits so weighted average matches shrunk aggregate
                mu_j_agg_raw = results['mu_J_up'] * p_up + results['mu_J_down'] * p_down
                shift = mu_j_new - mu_j_agg_raw
                fixed['mu_J_up'] = results['mu_J_up'] + shift
                fixed['mu_J_down'] = results['mu_J_down'] + shift
                
                # Scale sigma_J splits proportionally
                if results['sigma_J_up'] > 0 or results['sigma_J_down'] > 0:
                    sigma_j_agg_raw = np.sqrt(
                        results['sigma_J_up']**2 * p_up + 
                        results['sigma_J_down']**2 * p_down
                    )
                    if sigma_j_agg_raw > 1e-10:
                        scale = sigma_j_new / sigma_j_agg_raw
                        fixed['sigma_J_up'] = results['sigma_J_up'] * scale
                        fixed['sigma_J_down'] = results['sigma_J_down'] * scale
            else:
                fixed['lambda_up'] = priors['lambda_up_0']
                fixed['lambda_down'] = priors['lambda_down_0']
                fixed['mu_J_up'] = priors['mu_j_up_0']
                fixed['mu_J_down'] = priors['mu_j_down_0']
                fixed['sigma_J_up'] = priors['sigma_j_up_0']
                fixed['sigma_J_down'] = priors['sigma_j_down_0']
        
        fixed['rls_applied'] = True
        fixed['data_weight'] = extreme_data_weight if (mu_is_extreme or mu_j_is_extreme) else normal_data_weight
        fixed['prior_weight'] = extreme_prior_weight if (mu_is_extreme or mu_j_is_extreme) else normal_prior_weight
                

        # === Sign-correct the down-branch when n_jumps=0 and mu_J_down is positive ===
        if n_jumps == 0 and fixed.get('mu_J_down', 0) > 0:
            fixed['mu_J_down'] = -abs(fixed['mu_J_down'])
    
    # Align rates with prior ratio (up-jumps more common than down for risers)
            prior_up = priors.get('lambda_up_0', 0.5)
            prior_down = priors.get('lambda_down_0', 0.5)
            prior_total = prior_up + prior_down
            if prior_total > 0:
                prior_ratio = prior_up / prior_total
                lambda_total = fixed.get('lambda_up', 0) + fixed.get('lambda_down', 0)
                fixed['lambda_up'] = lambda_total * prior_ratio
                fixed['lambda_down'] = lambda_total * (1 - prior_ratio)
               
        if verbose:
            print(f"\n RLS RESULTS:")
            print(f"  μ: {results.get('mu', 0):.4f} → {fixed['mu']:.4f}")
            print(f"  σ: {results.get('sigma', 0):.4f} → {fixed['sigma']:.4f}")
            print(f"  μ_J: {results.get('mu_J', 0):.4f} → {fixed['mu_J']:.4f}")
            print(f"  σ_J: {results.get('sigma_J', 0):.4f} → {fixed['sigma_J']:.4f}")

            
        return fixed
    
    def estimate(self, 
                 log_returns: np.ndarray,
                 position: str,
                 age: Optional[int] = None,
                 current_value: Optional[float] = None,
                 hard_em_results: Optional[Dict] = None,
                 verbose: bool = False) -> Dict:
        """Full pipeline: Hard EM + RLS ONLY if needed."""
        
        from hard_em import HardEMJumpDiffusion
        
        n_obs = len(log_returns)
        
        if hard_em_results is None:
            hard_em = HardEMJumpDiffusion(max_iter=50, kmax=1)
            results = hard_em.fit(log_returns, dt=1.0, verbose=False)
        else:
            results = hard_em_results
        
        needs_rls, reasons = self.is_edge_case(results, n_obs, age)
        
        if needs_rls:
            if verbose:
                print(f"\n  EDGE CASE DETECTED: {', '.join(reasons)}")
            results = self.apply_rls(
                results, n_obs, position, age, current_value, verbose
            )
        else:
            if verbose:
                print(f"\n NORMAL CASE - No RLS needed")
            results['rls_applied'] = False
        
        return results


# === Usage ===
if __name__ == "__main__":
    rls = ConstrainedRLS('prior_db.pkl')
    
    from hard_em import HardEMJumpDiffusion
    
    log_returns = np.array([0.2, 0.3, 0.1, 0.4, 0.5, 0.2, 0.3, 0.1, 0.4, 0.5, 0.3])
    
    results = rls.estimate(
        log_returns=log_returns,
        position='Midfield',
        age=19,
        current_value=30_000_000,
        verbose=True
    )
