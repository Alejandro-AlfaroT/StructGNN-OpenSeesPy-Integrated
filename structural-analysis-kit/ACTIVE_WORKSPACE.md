# Active desktop research workspace

Generation preference: the user specifies five workers for the home desktop. Generation was explicitly stopped on 2026-09-10 using the scheduler's stop request for r150_s200. Do not restart generation without a subsequent request; use --workers 5 when a desktop restart is requested.

This local copy was established on 2026-09-10 (America/Los_Angeles) from the OneDrive analysis kit. The active task remains in the parent repository; use this folder for analysis scripts, profiles, and dated outputs. Generated data remain in ../RC Structure/outputs. The original OneDrive kit and its historical reports were not modified.

Read AGENTS.md, DATA_CONTRACT.md, and WORKFLOW.md for subsequent research requests. Use config/desktop.local.json on DROPC for desktop-local data. Its project root and user-supplied C:\Users\andro\anaconda3\envs\OpPy\python.exe have been verified; Python 3.12.13 ran scripts/preflight.py successfully. Optional surrogate_root remains null; the sibling RC Hybrid Surrogate Model exists, but checkpoints have not been validated.

For SSD analysis on this device, use config/desktop-ssd.local.json and SSD_WORKFLOW.md. The verified desktop interpreter is recorded there too; dataset_base remains unset and requires a confirmed SSD mount. No external volume was visible during setup. Use the laptop profiles only on the laptop after checking its own paths.

Requested task model: GPT-6 Astra. These JSON profiles do not set the Codex model. This setup did not verify or change the task's app model selector; select GPT-6 Astra there if necessary.

Latest readiness report: outputs/20260911T042005Z_desktop_ad04de2c/readiness.md. Preflight completed successfully. The earlier outputs/20260911T041406Z_desktop_setup_e1f59e92/report.md preserves initial coverage, candidate plans, metric risks and the review queue; its interpreter blocker has now been resolved.

To rerun preflight when needed, from this directory:

```powershell
$profile = Get-Content -Raw -LiteralPath config/desktop.local.json | ConvertFrom-Json
& $profile.python_executable -B scripts/preflight.py --profile config/desktop.local.json
```

Continue routine QA/EDA autonomously under WORKFLOW.md. Preserve raw inputs and manifests, tie metrics to source runs and definitions, retain solver failures and censoring separately from physical extremes and data defects, and establish plan provenance before admitting any new comparison set.
