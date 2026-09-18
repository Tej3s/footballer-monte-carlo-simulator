"""
check_player.py - Check if a player exists in your dataset.
"""

import pandas as pd
import numpy as np
# Load data
players = pd.read_csv('players.csv')
valuations = pd.read_csv('player_valuations.csv')

# ============================================
# SEARCH BY NAME
# ============================================
print("="*60)
print("SEARCHING FOR 'DIOMANDE'")
print("="*60)

diomande = players[players['name'].str.contains('Diomande', case=False, na=False)]
print(diomande[['player_id', 'name', 'position']])

# ============================================
# SEARCH BY ID
# ============================================
print("\n" + "="*60)
print("SEARCHING FOR ID 1390649")
print("="*60)

player_by_id = players[players['player_id'] == 1390649]
print(player_by_id[['player_id', 'name', 'position']])

# ============================================
# CHECK VALUATIONS
# ============================================
print("\n" + "="*60)
print("VALUATIONS FOR 1390649")
print("="*60)

if not player_by_id.empty:
    player_vals = valuations[valuations['player_id'] == 1390649]
    print(f"Number of valuations: {len(player_vals)}")
    if len(player_vals) > 0:
        print(player_vals[['date', 'market_value_in_eur']])

# ============================================
# SEARCH FOR SIMILAR NAMES
# ============================================
print("\n" + "="*60)
print("SEARCHING FOR SIMILAR NAMES")
print("="*60)

similar = players[players['name'].str.contains('Yan|Diom|Ivor', case=False, na=False)]
print(similar[['player_id', 'name', 'position']].head(20))

import pandas as pd

valuations = pd.read_csv('player_valuations.csv')
valuations['date'] = pd.to_datetime(valuations['date'])

# Check Diomande's valuations
diomande_vals = valuations[valuations['player_id'] == 1390649]

print(f"Number of valuations: {len(diomande_vals)}")
print("\nValuations:")
print(diomande_vals[['date', 'market_value_in_eur']].to_string())

# Check log returns (need at least 3)
if len(diomande_vals) >= 2:
    values = diomande_vals['market_value_in_eur'].values
    log_returns = []
    for i in range(1, len(values)):
        if values[i-1] > 0 and values[i] > 0:
            log_returns.append(np.log(values[i] / values[i-1]))
    print(f"\nLog returns: {len(log_returns)}")
    print(log_returns)
else:
    print("\n❌ Not enough valuations to compute log returns")


import pandas as pd

valuations = pd.read_csv('player_valuations.csv')
valuations['date'] = pd.to_datetime(valuations['date'])

# All of Fermín's valuations in the raw CSV
fermin = valuations[valuations['player_id'] == 636703].sort_values('date')
print("All valuations in CSV:")
print(fermin[['date', 'market_value_in_eur']].to_string())

# What passes the 2026-06-04 filter
cutoff = pd.Timestamp('2026-06-04')
filtered = fermin[fermin['date'] <= cutoff]
print(f"\nAfter filter (date <= {cutoff}):")
print(filtered[['date', 'market_value_in_eur']].to_string())
print(f"Count: {len(filtered)}")

import pandas as pd

valuations = pd.read_csv('player_valuations.csv')
valuations['date'] = pd.to_datetime(valuations['date'])

# Michael Olise's player_id — confirm below if unsure
olise_id = 566723

olise = valuations[valuations['player_id'] == olise_id].sort_values('date')

print(f"Total valuations: {len(olise)}")
print(f"First: {olise['date'].min()}")
print(f"Last:  {olise['date'].max()}")
print()
print(olise[['date', 'market_value_in_eur']].to_string(index=False))

vitinha_id = 487469
vitinha = valuations[valuations['player_id'] == vitinha_id].sort_values('date')
print(f"Total valuations: {len(vitinha)}")
print(f"First: {vitinha['date'].min()}")
print(f"Last:  {vitinha['date'].max()}")
print()
print(vitinha[['date', 'market_value_in_eur']].to_string(index=False))
