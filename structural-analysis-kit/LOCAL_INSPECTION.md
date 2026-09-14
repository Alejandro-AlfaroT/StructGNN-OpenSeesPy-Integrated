# Laptop inspection — 2026-09-10

The user confirmed this is the laptop and the useful generated case data are primarily on the desktop. Desktop paths are unknown until the user is home. Only the local device was available in this session. The user subsequently noted a new external SSD for carrying datasets between devices. It was not visible in the current filesystem drive listing, so the SSD profile is ready for its mount path and no transfer was attempted.

Observed source root: `C:/Users/andro/StructGNN/RC Structure`.

| Dataset | Local case directories | Manifest claims |
|---|---:|---|
| parameterized_2500 | 1001 | 2500 planned rows; 400 completed, 2100 pending |
| parameterized_expansion_2501_3800 | 100 | 1300 planned rows; 100 completed, 1200 pending |
| pilot30_s1 | 30 | 30 completed |
| pilot30_s2 | 30 | 30 completed |
| pilot30_s3 | 30 | 28 completed, 2 failed |

These counts are inventory observations, not verified usable-run totals. In particular, directory and manifest coverage differ for parameterized_2500. The first sampled existing old case was case_1500, while the first manifest row was a pending case_0001. Reconcile the full roster before making dataset-wide conclusions. Failed pilot rows were not diagnosed in this setup pass.

Found the pilot30 intensity-calibration artifact and generator support for calibrated intensity. No confirmed adaptive-sampling output root was identified in the inspected RC outputs top level. The ground-motion expansion is a separate experiment; it is not established to be an adaptive set.

The parent StructGNN folder also contains legacy `Data`, `Data_SAP2000`, `Results`, `GNN`, visualization utilities, and the newer `RC Hybrid Surrogate Model`. These should not be mixed indiscriminately. Scope future inspection to the roots selected in the profile.

Interpreter verified: `C:/Users/andro/anaconda3/envs/OpPy/python.exe`, Python 3.12.12. Installed distribution metadata: NumPy 2.4.2, SciPy 1.17.1, Matplotlib 3.10.8, OpenSeesPy 3.8.0.0, Torch 2.10.0+cu128, torch-geometric 2.7.0. Pandas, Seaborn and PyArrow were absent. Two selected NPZ files loaded with NumPy without pickle; no simulation was run.

Git HEAD at inspection: `0442f20765356aaab42a0f844538f72511ccb835`. Existing changes were an IDE project file and untracked root environment/plan files. No source repository files were changed by this setup.

Next useful action: on the desktop, fill the device profile, discover the actual active datasets, and run an inventory followed by manifest/run reconciliation. The reusable task is prepared for that step; the desktop itself has not been configured or inspected.
