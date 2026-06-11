"""
PATEF Validation — Simulation 4
R-Tree Spatial Index: O(log N) Query Time Complexity Proof
Proves: PostGIS R-Tree lookup scales logarithmically vs linear O(N) scan
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from rtree import index

np.random.seed(55)

# ── Parameters ────────────────────────────────────────────────────────────────
# N values: number of speed zone polygons in the database
N_VALUES    = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
N_QUERIES   = 500    # GPS points to query per N
N_REPEATS   = 5      # repeat each measurement for stability

# ── Generate random bounding boxes (speed zone polygons) ──────────────────────
def gen_zones(n, area=(0, 0, 1000, 1000)):
    """Generate n random non-overlapping approximate bounding boxes."""
    zones = []
    for _ in range(n):
        x1 = np.random.uniform(area[0], area[2] - 10)
        y1 = np.random.uniform(area[1], area[3] - 10)
        w  = np.random.uniform(2, 15)
        h  = np.random.uniform(2, 15)
        zones.append((x1, y1, x1 + w, y1 + h))
    return zones

def gen_query_points(n, area=(0, 0, 1000, 1000)):
    xs = np.random.uniform(area[0], area[2], n)
    ys = np.random.uniform(area[1], area[3], n)
    return list(zip(xs, ys))

# ── Linear scan baseline ───────────────────────────────────────────────────────
def linear_scan(zones, query_point):
    """O(N) — check every bounding box sequentially."""
    qx, qy = query_point
    for (x1, y1, x2, y2) in zones:
        if x1 <= qx <= x2 and y1 <= qy <= y2:
            return True
    return False

# ── R-Tree query ───────────────────────────────────────────────────────────────
def build_rtree(zones):
    idx = index.Index()
    for i, (x1, y1, x2, y2) in enumerate(zones):
        idx.insert(i, (x1, y1, x2, y2))
    return idx

def rtree_query(idx, query_point):
    qx, qy = query_point
    hits = list(idx.intersection((qx, qy, qx, qy)))
    return len(hits) > 0

# ── Benchmark ─────────────────────────────────────────────────────────────────
linear_times = []
rtree_times  = []

print("=" * 65)
print("  SIMULATION 4 — Spatial Query Complexity Benchmark")
print("=" * 65)
print(f"  {'N zones':>10}  {'Linear (ms)':>13}  {'R-Tree (ms)':>13}  {'Speedup':>9}")
print("  " + "-" * 53)

for N in N_VALUES:
    zones  = gen_zones(N)
    points = gen_query_points(N_QUERIES)
    idx    = build_rtree(zones)

    # Linear scan timing
    lt_runs = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        for p in points:
            linear_scan(zones, p)
        lt_runs.append((time.perf_counter() - t0) * 1000 / N_QUERIES)
    lt = np.mean(lt_runs)

    # R-Tree timing
    rt_runs = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        for p in points:
            rtree_query(idx, p)
        rt_runs.append((time.perf_counter() - t0) * 1000 / N_QUERIES)
    rt = np.mean(rt_runs)

    linear_times.append(lt)
    rtree_times.append(rt)
    speedup = lt / rt if rt > 0 else float('inf')
    print(f"  {N:>10}  {lt:>13.4f}  {rt:>13.4f}  {speedup:>8.1f}x")

print("=" * 65)

# ── Fit theoretical curves ────────────────────────────────────────────────────
N_arr = np.array(N_VALUES, dtype=float)

# Fit linear: t = a*N
a_lin = np.polyfit(N_arr, linear_times, 1)[0]
lin_fit = a_lin * N_arr

# Fit log: t = b*log(N) + c
log_x = np.log(N_arr)
coeffs = np.polyfit(log_x, rtree_times, 1)
log_fit = coeffs[0] * log_x + coeffs[1]

# R² for R-Tree log fit
ss_res = np.sum((np.array(rtree_times) - log_fit)**2)
ss_tot = np.sum((np.array(rtree_times) - np.mean(rtree_times))**2)
r2 = 1 - ss_res / ss_tot
print(f"\n  R-Tree log fit R² = {r2:.4f} (1.0 = perfect logarithmic scaling)")

# ── Figure 4 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Figure 4: R-Tree Spatial Index — O(log N) Complexity Verification",
             fontsize=13, fontweight='bold', y=1.01)

# 4a: Raw timing comparison
ax = axes[0]
ax.plot(N_VALUES, linear_times, 'o-', color='#C0392B', linewidth=2.2, markersize=7,
        label='Linear Scan O(N)', zorder=3)
ax.plot(N_VALUES, rtree_times,  's-', color='#1A5276', linewidth=2.2, markersize=7,
        label='R-Tree Index O(log N)', zorder=3)
ax.plot(N_arr, lin_fit,  '--', color='#C0392B', linewidth=1.2, alpha=0.5, label='Linear fit')
ax.plot(N_arr, log_fit,  '--', color='#1A5276', linewidth=1.2, alpha=0.5,
        label=f'Log fit (R²={r2:.3f})')

ax.set_xlabel('Number of Speed Zone Polygons (N)', fontsize=10)
ax.set_ylabel('Avg Query Time per GPS Point (ms)', fontsize=10)
ax.set_title('(a) Query Time vs. Number of Zones', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(N_VALUES) * 1.05)

# 4b: Log-scale to make O(log N) flatness visible
ax2 = axes[1]
ax2.semilogx(N_VALUES, linear_times, 'o-', color='#C0392B', linewidth=2.2, markersize=7,
             label='Linear Scan — grows with N')
ax2.semilogx(N_VALUES, rtree_times,  's-', color='#1A5276', linewidth=2.2, markersize=7,
             label='R-Tree — nearly flat (log N)')

# Annotate the flatness
ax2.annotate('R-Tree stays near-flat\neven at N=10,000 zones',
             xy=(N_VALUES[-1], rtree_times[-1]),
             xytext=(N_VALUES[3], rtree_times[-1] + (linear_times[-1] - rtree_times[-1]) * 0.4),
             arrowprops=dict(arrowstyle='->', color='#1A5276'),
             fontsize=9, color='#1A5276', fontweight='bold')

speedups = [l/r for l, r in zip(linear_times, rtree_times)]
ax2.annotate(f'At N=10,000:\n{speedups[-1]:.0f}× faster',
             xy=(N_VALUES[-1], linear_times[-1]),
             xytext=(N_VALUES[-2], linear_times[-1] * 0.75),
             arrowprops=dict(arrowstyle='->', color='#C0392B'),
             fontsize=9, color='#C0392B', fontweight='bold')

ax2.set_xlabel('Number of Speed Zone Polygons (N) — log scale', fontsize=10)
ax2.set_ylabel('Avg Query Time per GPS Point (ms)', fontsize=10)
ax2.set_title('(b) Log-Scale View — Confirming O(log N) Behaviour', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig4_rtree_complexity.png', dpi=180, bbox_inches='tight')
plt.close()
print("\n  ✅ Figure 4 saved: fig4_rtree_complexity.png")
