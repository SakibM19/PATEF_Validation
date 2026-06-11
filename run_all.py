"""
PATEF — Run All Simulations (Fixed for MSYS2/Windows Environment)
=================================================================
"""
import os
import sys
import subprocess

print("=" * 65)
print("  PATEF — Complete Validation Suite")
print("  Privacy-Preserving Automated Traffic Enforcement Framework")
print("=" * 65)
print()

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

simulations = [
    ("sim1_tot_filter.py",   "Simulation 1 — ToT Filter (False Positive Reduction)"),
    ("sim2_section_speed.py","Simulation 2 — Section Speed Control (Kangaroo Evasion)"),
    ("sim3_zscore.py",       "Simulation 3 — Z-score Anomaly Detection"),
    ("sim4_rtree.py",        "Simulation 4 — R-Tree O(log N) Complexity"),
    ("sim5_real_data.py",    "Simulation 5 — Real Beijing GPS Data Validation"),
]

for i, (script, label) in enumerate(simulations, 1):
    print(f"\n{'─'*65}")
    print(f"  Running {label}")
    print(f"{'─'*65}")
    
    # Using subprocess avoids spawning a broken cmd.exe shell context
    result = subprocess.run([sys.executable, script])
    
    if result.returncode != 0:
        print(f"\n  x {script} failed with exit code {result.returncode}")
        sys.exit(1)

print()
print("=" * 65)
print("  All simulations completed successfully!")
print("=" * 65)