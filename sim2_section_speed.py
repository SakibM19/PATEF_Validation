"""
PATEF Validation — Simulation 2
Section Speed Control (V_avg) vs Fixed Point Camera
Proves: Average speed enforcement catches checkpoint-evaders that fixed cameras miss
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

# ── Parameters ────────────────────────────────────────────────────────────────
SEGMENT_KM    = 10.0    # km — monitored highway segment
SPEED_LIMIT   = 80.0    # km/h
CHECKPOINT_KM = 5.0     # fixed camera at midpoint
N_VEHICLES    = 600

# ── Simulate each vehicle's journey ───────────────────────────────────────────
def simulate_journey(profile):
    """
    Returns (entry_time, exit_time, checkpoint_speed, avg_speed_kmh)
    profile:
      'compliant'   — drives at ~75 km/h throughout
      'kangaroo'    — drives fast but brakes near checkpoint
      'genuine'     — genuinely fast throughout
    """
    if profile == 'compliant':
        # Steady ~75 km/h
        cruise = np.random.uniform(68, 79)
        travel_h = SEGMENT_KM / cruise
        cp_speed = np.random.uniform(68, 79)

    elif profile == 'kangaroo':
        # Drives at 110–130 km/h but slows to ~70 near the camera
        fast_speed = np.random.uniform(108, 132)
        # Spend 80% of distance at fast speed, 20% braking/cruising near camera
        dist_fast = SEGMENT_KM * 0.80
        dist_slow = SEGMENT_KM * 0.20
        t_fast = dist_fast / fast_speed      # hours
        slow_speed = np.random.uniform(62, 78)
        t_slow = dist_slow / slow_speed
        travel_h = t_fast + t_slow
        cp_speed = slow_speed                # camera sees the slow speed

    elif profile == 'genuine':
        # Drives fast the whole way — no evasion
        cruise = np.random.uniform(95, 125)
        travel_h = SEGMENT_KM / cruise
        cp_speed = cruise + np.random.normal(0, 3)

    avg_speed = SEGMENT_KM / travel_h    # km/h
    return cp_speed, avg_speed

# ── Run ────────────────────────────────────────────────────────────────────────
data = {}
for profile in ['compliant', 'kangaroo', 'genuine']:
    data[profile] = [simulate_journey(profile) for _ in range(N_VEHICLES)]

def detection_rates(profile_data):
    cp_caught  = sum(1 for cp, _ in profile_data if cp   > SPEED_LIMIT)
    avg_caught = sum(1 for _,  av in profile_data if av  > SPEED_LIMIT)
    n = len(profile_data)
    return cp_caught / n * 100, avg_caught / n * 100

comp_cp,  comp_avg  = detection_rates(data['compliant'])
kang_cp,  kang_avg  = detection_rates(data['kangaroo'])
gen_cp,   gen_avg   = detection_rates(data['genuine'])

print("=" * 60)
print("  SIMULATION 2 — Section Speed Control Results")
print("=" * 60)
print(f"  Compliant vehicles falsely flagged:")
print(f"    Fixed camera : {comp_cp:.1f}%   |  PATEF V_avg : {comp_avg:.1f}%")
print(f"  Kangaroo evaders detected:")
print(f"    Fixed camera : {kang_cp:.1f}%   |  PATEF V_avg : {kang_avg:.1f}%")
print(f"  Genuine speeders detected:")
print(f"    Fixed camera : {gen_cp:.1f}%  |  PATEF V_avg : {gen_avg:.1f}%")
print("=" * 60)

# ── Figure 2 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Figure 2: Section Speed Control — PATEF V_avg vs Fixed Point Camera",
             fontsize=13, fontweight='bold', y=1.01)

# 2a: Detection rate bars
ax = axes[0]
cats   = ['Compliant\n(No violation)', 'Kangaroo Evaders\n(Brake at camera)', 'Genuine Speeders\n(Fast throughout)']
cp_r   = [comp_cp,  kang_cp,  gen_cp]
avg_r  = [comp_avg, kang_avg, gen_avg]
x = np.arange(len(cats))
w = 0.32

b1 = ax.bar(x - w/2, cp_r,  w, label='Fixed Point Camera', color='#C0392B', alpha=0.88, edgecolor='white')
b2 = ax.bar(x + w/2, avg_r, w, label='PATEF V_avg (Section)', color='#1A5276', alpha=0.88, edgecolor='white')

for bar, val in zip(list(b1) + list(b2), cp_r + avg_r):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='#C0392B' if bar in b1 else '#1A5276')

ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Detection Rate (%)', fontsize=10)
ax.set_title('(a) Detection Rate by Driver Profile', fontsize=11, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2b: Speed profile illustration of kangaroo evasion
ax2 = axes[1]
dist = np.linspace(0, SEGMENT_KM, 200)

# Kangaroo profile: fast, then slows near camera, then fast again
kang_speed = np.ones(200) * 118
kang_speed[80:110] = np.linspace(118, 65, 30)
kang_speed[110:130] = np.linspace(65, 70, 20)
kang_speed[130:160] = np.linspace(70, 115, 30)

# Compliant profile
comp_speed = np.ones(200) * 74 + np.random.normal(0, 1.5, 200)

ax2.plot(dist, kang_speed, color='#E67E22', linewidth=2.2, label='Kangaroo Evader')
ax2.plot(dist, comp_speed, color='#27AE60', linewidth=2.2, label='Compliant Driver', alpha=0.85)
ax2.axhline(y=SPEED_LIMIT, color='#C0392B', linestyle='--', linewidth=1.5, label=f'Speed Limit ({SPEED_LIMIT} km/h)')
ax2.axvline(x=CHECKPOINT_KM, color='purple', linestyle=':', linewidth=2, label='Fixed Camera Position')

ax2.fill_between(dist, SPEED_LIMIT, kang_speed,
                 where=(kang_speed > SPEED_LIMIT),
                 alpha=0.12, color='red', label='Excess speed zone')

# V_avg annotation
v_avg_kang = SEGMENT_KM / sum([(kang_speed[i] if kang_speed[i] > 0 else 1)
                                 for i in range(200)]) * 200 / 3600 * 1000
# simpler: use mean
v_avg_val = np.mean(kang_speed)
ax2.annotate(f'V_avg = {v_avg_val:.0f} km/h\n→ PATEF cites',
             xy=(9.5, v_avg_val), xytext=(6.5, 108),
             arrowprops=dict(arrowstyle='->', color='#1A5276'),
             fontsize=9, color='#1A5276', fontweight='bold')

ax2.annotate(f'Camera sees\n{65:.0f} km/h\n→ No citation',
             xy=(CHECKPOINT_KM, 65), xytext=(6.2, 45),
             arrowprops=dict(arrowstyle='->', color='purple'),
             fontsize=9, color='purple')

ax2.set_xlabel('Distance Along Segment (km)', fontsize=10)
ax2.set_ylabel('Speed (km/h)', fontsize=10)
ax2.set_title('(b) Kangaroo Evasion — Camera vs PATEF V_avg', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8.5, loc='lower right')
ax2.set_ylim(30, 130)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig2_section_speed.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  ✅ Figure 2 saved: fig2_section_speed.png")
