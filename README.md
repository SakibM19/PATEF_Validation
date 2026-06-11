# PATEF — Privacy-Preserving Automated Traffic Enforcement Framework

PATEF is an automated framework designed for traffic enforcement. This repository contains the complete validation and simulation suite used to test framework capabilities such as false positive reduction, anomaly detection, and scaling efficiency.

## 📂 Project Structure

- `run_all.py` — The main orchestrator script that runs the entire validation suite.
- `sim1_tot_filter.py` — Simulation 1: Time-over-Threshold (ToT) filter testing for false-positive reduction against GPS drift.
- `sim2_section_speed.py` — Simulation 2: Section speed control vs. fixed-point camera enforcement.
- `sim3_zscore.py` — Simulation 3: Z-score anomaly detection to find reckless drivers in heavy congestion.
- `sim4_rtree.py` — Simulation 4: O(log N) query time complexity proof using an R-Tree spatial index.
- `sim5_real_data.py` — Simulation 5: Validation using real-world Microsoft GeoLife-characteristic Beijing GPS data.
- `beijing_gps_data.csv` — The dataset containing anonymized vehicle trajectories used for Simulation 5.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib scipy rtree