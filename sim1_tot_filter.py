"""
PATEF Validation — Simulation 1
Time-over-Threshold (ToT) Filter vs Naive Instant-Trigger System
Proves: ToT eliminates GPS drift false positives while catching real violations
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ── Parameters ────────────────────────────────────────────────────────────────
SPEED_LIMIT     = 60.0      # km/h
TOT_THRESHOLD   = 5.0       # seconds — must exceed limit for this long
SAMPLE_RATE     = 1.0       # GPS reading every 1 second
N_VEHICLES      = 500       # synthetic vehicles
SIM_DURATION    = 120       # seconds of GPS trace per vehicle

# ── Synthetic GPS trace generator ─────────────────────────────────────────────
def generate_trace(vehicle_type):
    """
    vehicle_type:
      'compliant'  — drives at 50 km/h with small noise
      'gps_drift'  — drives at 50 km/h but has 1-3s random spikes > 60
      'speeder'    — genuinely exceeds 60 km/h for 8-20s continuously
    """
    t = np.arange(0, SIM_DURATION, SAMPLE_RATE)
    speed = np.zeros(len(t))

    if vehicle_type == 'compliant':
        speed = np.random.normal(50, 3, len(t))
        speed = np.clip(speed, 30, 59)

    elif vehicle_type == 'gps_drift':
        base = np.random.normal(50, 3, len(t))
        base = np.clip(base, 30, 58)
        # inject 1–3 random spike windows of 1–2 seconds
        n_spikes = np.random.randint(1, 4)
        for _ in range(n_spikes):
            spike_start = np.random.randint(0, len(t) - 3)
            spike_len   = np.random.randint(1, 3)   # SHORT — 1 or 2 seconds
            base[spike_start:spike_start + spike_len] = np.random.uniform(75, 120)
        speed = base

    elif vehicle_type == 'speeder':
        base = np.random.normal(50, 3, len(t))
        base = np.clip(base, 30, 59)
        # inject 1 genuine sustained violation of 8–20 seconds
        vio_start = np.random.randint(10, len(t) - 25)
        vio_len   = np.random.randint(8, 21)         # LONG — sustained
        base[vio_start:vio_start + vio_len] = np.random.uniform(70, 110)
        speed = base

    return t, np.clip(speed, 0, 200)

# ── Detection logic ────────────────────────────────────────────────────────────
def naive_detect(speed_trace):
    """Instant-trigger: flag if ANY single reading exceeds limit."""
    return int(np.any(speed_trace > SPEED_LIMIT))

def tot_detect(speed_trace, threshold_secs=TOT_THRESHOLD, sample_rate=SAMPLE_RATE):
    """ToT filter: flag only if limit exceeded CONTINUOUSLY for >= threshold."""
    threshold_samples = int(threshold_secs / sample_rate)
    consecutive = 0
    for v in speed_trace:
        if v > SPEED_LIMIT:
            consecutive += 1
            if consecutive >= threshold_samples:
                return 1
        else:
            consecutive = 0
    return 0

# ── Run simulation ─────────────────────────────────────────────────────────────
results = {'compliant': [], 'gps_drift': [], 'speeder': []}

for vtype in results:
    for _ in range(N_VEHICLES):
        _, spd = generate_trace(vtype)
        naive = naive_detect(spd)
        tot   = tot_detect(spd)
        results[vtype].append((naive, tot))

def rates(res_list):
    naive_flags = sum(r[0] for r in res_list)
    tot_flags   = sum(r[1] for r in res_list)
    n = len(res_list)
    return naive_flags / n * 100, tot_flags / n * 100

comp_naive,  comp_tot  = rates(results['compliant'])
drift_naive, drift_tot = rates(results['gps_drift'])
speed_naive, speed_tot = rates(results['speeder'])

print("=" * 55)
print("  SIMULATION 1 — ToT Filter Performance Results")
print("=" * 55)
print(f"  Compliant vehicles falsely flagged:")
print(f"    Naive system : {comp_naive:.1f}%")
print(f"    PATEF ToT    : {comp_tot:.1f}%")
print(f"  GPS-drift vehicles falsely flagged:")
print(f"    Naive system : {drift_naive:.1f}%")
print(f"    PATEF ToT    : {drift_tot:.1f}%")
print(f"  True speeders correctly detected:")
print(f"    Naive system : {speed_naive:.1f}%")
print(f"    PATEF ToT    : {speed_tot:.1f}%")
print("=" * 55)

# ── Figure 1a: Bar chart comparison ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Figure 1: ToT Filter Performance — PATEF vs Naive Instant-Trigger System",
             fontsize=13, fontweight='bold', y=1.01)

categories  = ['Compliant\n(No violation)', 'GPS Drift\n(False spikes)', 'True Speeders\n(Genuine violation)']
naive_vals  = [comp_naive,  drift_naive,  speed_naive]
tot_vals    = [comp_tot,    drift_tot,    speed_tot]

x = np.arange(len(categories))
w = 0.32

ax = axes[0]
b1 = ax.bar(x - w/2, naive_vals, w, label='Naive Instant-Trigger', color='#C0392B', alpha=0.88, edgecolor='white')
b2 = ax.bar(x + w/2, tot_vals,   w, label='PATEF ToT Filter',      color='#1A5276', alpha=0.88, edgecolor='white')

for bar in b1:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, color='#C0392B', fontweight='bold')
for bar in b2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, color='#1A5276', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9.5)
ax.set_ylabel('Citation Rate (%)', fontsize=10)
ax.set_title('(a) Citation Rate by Vehicle Type', fontsize=11, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='black', linewidth=0.5)

# ── Figure 1b: Example GPS trace showing drift vs real violation ───────────────
ax2 = axes[1]
t_ex = np.arange(0, 60, 1)

# Drifting vehicle
np.random.seed(7)
drift_trace = np.random.normal(48, 2.5, len(t_ex))
drift_trace[18:20] = [88, 92]   # 2-second spike
drift_trace[40:42] = [95, 78]   # another 2-second spike
drift_trace = np.clip(drift_trace, 0, 130)

# Real speeder
np.random.seed(99)
speed_trace = np.random.normal(48, 2.5, len(t_ex))
speed_trace[25:38] = np.random.uniform(72, 95, 13)   # 13-second violation
speed_trace = np.clip(speed_trace, 0, 130)

ax2.plot(t_ex, drift_trace,  color='#E67E22', linewidth=1.8, label='GPS Drift Vehicle',   alpha=0.9)
ax2.plot(t_ex, speed_trace,  color='#1A5276', linewidth=1.8, label='True Speeder',         alpha=0.9)
ax2.axhline(y=SPEED_LIMIT,   color='#C0392B', linewidth=1.5, linestyle='--', label=f'Speed Limit ({SPEED_LIMIT} km/h)')
ax2.axhspan(0, SPEED_LIMIT,  alpha=0.04, color='green')
ax2.axhspan(SPEED_LIMIT, 130,alpha=0.04, color='red')

ax2.annotate('GPS spike\n(1–2s) → PATEF ignores',
             xy=(19, 92), xytext=(28, 108),
             arrowprops=dict(arrowstyle='->', color='#E67E22'),
             fontsize=8.5, color='#E67E22')
ax2.annotate('Sustained violation\n(13s) → PATEF cites',
             xy=(31, 85), xytext=(3, 108),
             arrowprops=dict(arrowstyle='->', color='#1A5276'),
             fontsize=8.5, color='#1A5276')

ax2.set_xlabel('Time (seconds)', fontsize=10)
ax2.set_ylabel('Vehicle Speed (km/h)', fontsize=10)
ax2.set_title('(b) Sample GPS Traces: Drift vs Genuine Violation', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8.5, loc='lower right')
ax2.set_ylim(0, 130)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig1_tot_filter.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  ✅ Figure 1 saved: fig1_tot_filter.png")
