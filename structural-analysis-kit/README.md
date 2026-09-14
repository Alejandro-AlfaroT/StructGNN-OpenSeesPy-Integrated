# Structural research analysis workspace

Prepared 2026-09-10. The desktop is the primary source of generated research data. This laptop can also analyze datasets carried on the new external SSD. Its existing partial copies do not establish desktop coverage or current run health. For the portable-data setup, start with SSD_WORKFLOW.md and config/ssd.local.json; fill the mounted dataset_base when the SSD is connected.

## Start on the desktop

1. Copy this entire `structural-analysis-kit` folder to a local analysis folder outside the generator's outputs. Keep the large datasets where they already live.
2. Open that folder as a local Codex project. Select **GPT-6 Astra**, with **High** reasoning as a starting preference. Model selection is an app setting; the JSON files below describe the analysis workflow and do not change the model.
3. Edit `config/desktop.local.json`: set the actual `project_root` and `python_executable`. The project root should contain `Structure_Parameters.py`, `Data_Generation`, and `outputs` (normally the `RC Structure` directory). Set `surrogate_root` if you want surrogate evaluation. Confirm each dataset's relative path; add the actual calibrated/adaptive dataset roots when discovered. Their locations are currently unknown.
4. Use the prompt in `TASK_PROMPT.md`, with `config/desktop.local.json` as the profile. The first task is a bounded inventory and readiness check. Then use one of the analysis requests in `WORKFLOW.md`.

The desktop repository path is now recorded as `C:\Users\andro\Documents\GitHub\StructGNN-OpenSeesPy-Integrated`. The desktop profiles propose its `RC Structure` subfolder based on the laptop layout; verify that folder on the desktop. The desktop Python interpreter remains unset. Use `config/desktop-ssd.local.json` for SSD data on the desktop; its data mount path remains unset. `config/ssd.local.json` remains the laptop SSD profile. Do not copy the laptop's absolute paths into it without checking them. No desktop connection, automatic synchronization, or recurring monitor has been configured.

## Verify locally

From this folder, using an existing Python 3.10+ interpreter:

```text
python -B scripts/preflight.py --profile config/desktop.local.json
```

On the inspected laptop:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' -B scripts/preflight.py --profile config/laptop.local.json
```

Preflight uses the Python standard library. It inventories configured roots, counts case directories and manifest statuses, hashes the small source files it reads, samples metadata, and records the interpreter/package versions. It creates a fresh `outputs/<UTC>_<device>_<id>/` report. It does not certify datasets or run OpenSees. Existing directories are not proof of complete runs. The scan is not an atomic snapshot of a live generator.

The laptop's OpPy environment has NumPy, SciPy, Matplotlib, OpenSeesPy, and Torch. Pandas, Seaborn, and PyArrow were not installed in that interpreter. They are optional for setup; use existing NumPy/CSV tools or install needed analysis packages in a separate environment later. `requirements-analysis.txt` records the three installed analysis-library versions checked here. GPU/Torch/OpenSees are unnecessary for ordinary metadata and response analysis.

## Files to reuse

| File/folder | Purpose |
|---|---|
| `AGENTS.md` | Persistent research and QA instructions |
| `TASK_PROMPT.md` | First message for a new local task |
| `config/*.local.json` | Paths and dataset selection for each device |
| `SSD_WORKFLOW.md` | Portable data, per-device paths, and verified batch transfers |
| `WORKFLOW.md` | Repeatable inventory → QA → EDA → comparison → export workflow |
| `DATA_CONTRACT.md` | Actual files, metric definitions, schema differences, and pitfalls found locally |
| `LOCAL_INSPECTION.md` | What was inspected on this laptop and what remains unknown |
| `scripts/` | Reusable analysis helpers |
| `work/` | Temporary calculations and cached working tables |
| `outputs/` | Dated, reviewable analysis products and provenance |

Keep code, shared rules, and generic profiles together. Keep each device's absolute paths local. For a Git-managed copy, `.gitignore` excludes local profiles, working data, and generated outputs; distribute a profile example or this initial package separately. Never use two devices to write the same live output folder. To review desktop work on the laptop, transfer a completed report folder with its provenance and required compact tables. Do not overwrite generator state/manifests while transferring results.

Codex uses `AGENTS.md` for project instructions. Local environment actions can be configured in the app and stored under `.codex`; this kit does not invent an app environment-file schema. An optional app action can run the preflight command above. [Project instructions](https://learn.chatgpt.com/docs/agent-configuration/agents-md), [local environments](https://learn.chatgpt.com/docs/environments/local-environment).

GPT-6 Astra is the requested model; availability must be checked on the destination device/account. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra).
