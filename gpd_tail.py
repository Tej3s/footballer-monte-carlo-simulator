
import numpy as np
import pickle
import os
from scipy.stats import genpareto


GPD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpd_params.pkl')


def fit_gpd_tail(all_log_returns, threshold_percentile=82, verbose=True):
    
    x = np.asarray(all_log_returns, dtype=float)
    x = x[np.isfinite(x)]
    
    if len(x) < 100:
        raise ValueError(f"Only {len(x)} log-returns. Need at least 100.")
    
    threshold = float(np.percentile(x, threshold_percentile))
    exceedances = x[x > threshold] - threshold
    
    if len(exceedances) < 50:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold={threshold:.4f}. "
            f"Lower the percentile."
        )
    
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    
    params = {
        'threshold': threshold,
        'shape': float(shape),
        'scale': float(scale),
        'n_exceedances': int(len(exceedances)),
        'n_total': int(len(x)),
        'threshold_percentile': float(threshold_percentile),
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("GPD TAIL FIT")
        print(f"{'='*60}")
        print(f"Total log-returns:      {params['n_total']}")
        print(f"Threshold ({threshold_percentile}th pct): {threshold:.4f}")
        print(f"Exceedances:            {params['n_exceedances']}")
        print(f"Shape (xi):             {params['shape']:.4f}")
        print(f"Scale (beta):           {params['scale']:.4f}")
        
        if params['shape'] < 0:
            upper = threshold - scale / shape
            print(f"\nξ < 0 → BOUNDED TAIL")
            print(f"  Max log-return:  {upper:.4f}")
            print(f"  Max multiplier:  {np.exp(upper):.2f}x")
        elif abs(params['shape']) < 0.01:
            print(f"\nξ ≈ 0 → EXPONENTIAL TAIL")
            print(f"  Decay rate: exp(-x/{scale:.4f})")
        else:
            print(f"\nξ > 0 → HEAVY TAIL")
            print(f"  GPD will not bound the tail. Consider truncated GPD or cap.")
    
    return params


def save_gpd(params, path=GPD_PATH):
    with open(path, 'wb') as f:
        pickle.dump(params, f)
    print(f"\nSaved GPD params to: {os.path.abspath(path)}")


def load_gpd(path=GPD_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GPD params not found at {path}. "
            f"Run `python gpd_tail.py` first."
        )
    with open(path, 'rb') as f:
        return pickle.load(f)


def sample_excess(params, n=1):
    """Draw n samples from the fitted GPD (positive excess values)."""
    return genpareto.rvs(
        params['shape'], loc=0, scale=params['scale'], size=n
    )



if __name__ == "__main__":
    import sys
    from hard_em import get_player_log_returns, HardEMJumpDiffusion
    
    print("Loading player log-returns...")
    all_players = get_player_log_returns(min_valuations = 2)
    print(f"Loaded {len(all_players)} players")

    #collecting jump returns
    all_jump_returns = []
    hem = HardEMJumpDiffusion(max_iter=50, kmax=1)

    for i, (_,row) in enumerate(all_players.iterrows()):
        lr = np.asarray(row['log_returns'], dtype = float)
        if len(lr) < 5:
            continue

        try:
            res = hem.fit(lr, dt=1.0, verbose = False)
        except Exception:
            continue

        for idx in res['jump_indices']:
            all_jump_returns.append(lr[idx])

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(all_players)} players, "
                  f"{len(all_jump_returns)} jumps collected")

    all_jump_returns = np.array(all_jump_returns)
    print(f"\nTotal jump returns: {len(all_jump_returns)}")
    
    
    
    # Fit
    params = fit_gpd_tail(all_jump_returns, threshold_percentile=82)
    
    # Save
    save_gpd(params)