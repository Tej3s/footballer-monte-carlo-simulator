from hard_em import get_player_log_returns, HardEMJumpDiffusion
import numpy as np
from scipy.stats import genpareto

all_players = get_player_log_returns(min_valuations=10)  # only well-fit players

all_jump_returns = []
he = HardEMJumpDiffusion(max_iter=50, kmax=1)
for _, row in all_players.iterrows():
    lr = np.array(row['log_returns'])
    if len(lr) < 5:
        continue
    res = he.fit(lr, dt=1.0, verbose=False)
    for idx in res['jump_indices']:
        all_jump_returns.append(lr[idx])

all_jump_returns = np.array(all_jump_returns)
print(f"Pooled jump returns: {len(all_jump_returns)}")
print(f"90th percentile of jumps: {np.percentile(all_jump_returns, 90):.4f}")
print(f"96th percentile of jumps: {np.percentile(all_jump_returns, 96):.4f}")

for pct in [70, 75, 80, 82, 85, 88, 90, 92]:
    threshold = np.percentile(all_jump_returns, pct)
    exceedances = all_jump_returns[all_jump_returns > threshold] - threshold
    if len(exceedances) < 100:
        continue
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    tag = "BOUNDED" if shape < 0 else "unbounded"
    print(f"{pct}th: thr={threshold:.4f}, n={len(exceedances):6d}, ξ={shape:+.4f}, β={scale:.4f}  [{tag}]")