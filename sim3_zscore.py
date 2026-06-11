"""
PATEF Validation — Simulation 3
Z-score Anomaly Detection — Reckless Driving Under Congestion
Proves: Z-score correctly isolates dangerous outliers that fixed-limit systems miss
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

np.random.seed(21)

# ── Parameters ────────────────────────────────────────────────────────────────
SPEED_LIMIT         = 80.0   # km/h — posted highway limit
CONGESTION_THRESH   = 30.0   # km/h — mean below this = congested
Z_THRESHOLD         = 3.0    # anomaly flag threshold
N_SCENARIOS         = 5      # different congestion levels to test
N_VEHICLES          = 100    # vehicles per scenario

# ── Scenario definitions ───────────────────────────────────────────────────────
scenarios = [
    {'label': 'Severe\nCongestion',   'mu': 10,  'sigma': 2.5},
    {'label': 'Heavy\nCongestion',    'mu': 18,  'sigma': 3.5},
    {'label': 'Moderate\nCongestion', 'mu': 28,  'sigma': 5.0},
    {'label': 'Light\nCongestion',    'mu': 42,  'sigma': 6.5},
    {'label': 'Free\nFlow',           'mu': 68,  'sigma': 8.0},
]

# Reckless driver consistently drives at mu + 4*sigma
results = []
print("=" * 70)
print("  SIMULATION 3 — Z-score Anomaly Detection Results")
print("=" * 70)
print(f"  {'Scenario':<22} {'mu':>6} {'sigma':>7} {'V_reckless':>12} {'Z-score':>9} {'Fixed Flag':>11} {'PATEF Flag':>11}")
print("  " + "-" * 68)

for sc in scenarios:
    mu    = sc['mu']
    sigma = sc['sigma']
    # generate traffic fleet
    fleet = np.random.normal(mu, sigma, N_VEHICLES)
    fleet = np.clip(fleet, 0, SPEED_LIMIT + 20)
    # reckless vehicle
    v_reckless = mu + 4.0 * sigma
    # recalculate mu/sigma from fleet (as server would)
    real_mu    = np.mean(fleet)
    real_sigma = np.std(fleet)
    z = (v_reckless - real_mu) / real_sigma
    fixed_flag = int(v_reckless > SPEED_LIMIT)
    patef_flag = int(z > Z_THRESHOLD and real_mu < CONGESTION_THRESH)
    results.append({**sc, 'v_reckless': v_reckless, 'z': z,
                    'fixed': fixed_flag, 'patef': patef_flag,
                    'real_mu': real_mu, 'real_sigma': real_sigma})
    flag_f = '✓ FLAGGED' if fixed_flag else '✗ MISSED'
    flag_p = '✓ FLAGGED' if patef_flag else '✗ MISSED'
    print(f"  {sc['label'].replace(chr(10),' '):<22} {mu:>6.0f} {sigma:>7.1f} "
          f"{v_reckless:>12.1f} {z:>9.2f} {flag_f:>11} {flag_p:>11}")

print("=" * 70)

# ── Figure 3 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Figure 3: Z-Score Anomaly Detection — Reckless Driving Under Congestion",
             fontsize=13, fontweight='bold', y=1.01)

# 3a: Z-scores across scenarios
ax = axes[0]
labels   = [r['label'] for r in results]
z_vals   = [r['z']     for r in results]
colors   = ['#C0392B' if r['z'] > Z_THRESHOLD else '#2980B9' for r in results]

bars = ax.bar(range(len(results)), z_vals, color=colors, alpha=0.88, edgecolor='white', width=0.55)
ax.axhline(y=Z_THRESHOLD, color='#C0392B', linestyle='--', linewidth=1.8,
           label=f'Z-score threshold = {Z_THRESHOLD}')
ax.axhline(y=0, color='black', linewidth=0.5)

for i, (bar, z) in enumerate(zip(bars, z_vals)):
    ax.text(bar.get_x() + bar.get_width()/2, z + 0.08,
            f'Z={z:.2f}', ha='center', va='bottom', fontsize=9,
            fontweight='bold', color=colors[i])

ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Z-Score of Reckless Vehicle', fontsize=10)
ax.set_title('(a) Z-Score Across Traffic Conditions', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

flagged_patch  = mpatches.Patch(color='#C0392B', alpha=0.88, label='Z > 3.0 → PATEF Alert')
normal_patch   = mpatches.Patch(color='#2980B9', alpha=0.88, label='Z ≤ 3.0 → No alert')
ax.legend(handles=[flagged_patch, normal_patch,
                   plt.Line2D([0],[0], color='#C0392B', linestyle='--', linewidth=1.8,
                              label=f'Z threshold = {Z_THRESHOLD}')],
          fontsize=8.5)

# 3b: Gaussian distribution for severe congestion scenario
ax2 = axes[1]
sc_severe = results[0]
mu_s  = sc_severe['real_mu']
sig_s = sc_severe['real_sigma']
vr_s  = sc_severe['v_reckless']
z_s   = sc_severe['z']

x_range = np.linspace(max(0, mu_s - 5*sig_s), mu_s + 5.5*sig_s, 400)
y_norm  = stats.norm.pdf(x_range, mu_s, sig_s)

ax2.plot(x_range, y_norm, color='#2980B9', linewidth=2.5, label='Traffic Speed Distribution')
ax2.fill_between(x_range, y_norm, where=(x_range <= mu_s + Z_THRESHOLD * sig_s),
                 alpha=0.18, color='#2980B9')
ax2.fill_between(x_range, y_norm, where=(x_range > mu_s + Z_THRESHOLD * sig_s),
                 alpha=0.35, color='#C0392B', label=f'Z > {Z_THRESHOLD} zone (0.013%)')

ax2.axvline(x=vr_s, color='#C0392B', linewidth=2.5, linestyle='-',
            label=f'Reckless vehicle: {vr_s:.1f} km/h (Z={z_s:.1f})')
ax2.axvline(x=SPEED_LIMIT, color='#E67E22', linewidth=1.8, linestyle='--',
            label=f'Posted limit: {SPEED_LIMIT} km/h')
ax2.axvline(x=mu_s, color='#27AE60', linewidth=1.5, linestyle=':',
            label=f'Traffic mean: μ = {mu_s:.1f} km/h')

ax2.annotate(f'Fixed limit\nmisses this\nvehicle\n({vr_s:.0f} < {SPEED_LIMIT} km/h)',
             xy=(SPEED_LIMIT + 0.3, max(y_norm)*0.3),
             xytext=(SPEED_LIMIT + 4, max(y_norm)*0.55),
             arrowprops=dict(arrowstyle='->', color='#E67E22'),
             fontsize=8.5, color='#E67E22')

ax2.set_xlabel('Vehicle Speed (km/h)', fontsize=10)
ax2.set_ylabel('Probability Density', fontsize=10)
ax2.set_title(f'(b) Speed Distribution — Severe Congestion (μ={mu_s:.1f} km/h)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8.2, loc='upper right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig3_zscore_anomaly.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  ✅ Figure 3 saved: fig3_zscore_anomaly.png")
