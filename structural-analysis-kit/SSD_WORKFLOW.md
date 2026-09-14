# Carrying the datasets on an external SSD

The desktop generates the main research data. The SSD can carry stable copies for analysis on the laptop or desktop. No SSD volume was visible in the drive listing during this setup, so no data were copied or written to an external drive.

## Suggested layout

```text
<SSD mount>/StructuralResearch/
  datasets/
    parameterized_2500/
    parameterized_expansion_2501_3800/
    pilot30_s1/
    pilot30_s2/
    pilot30_s3/
    <actual calibrated/adaptive roots>/
  dataset_manifests/          transfer rosters and content fingerprints
  analysis_kit/               copy of this reusable kit
  reports/<analysis_id>/     completed reports to carry between devices
```

The folders are a proposed destination layout, not directories already created on your SSD. Retain each dataset's plan, manifests, case directories and relative hierarchy when copying it. Keep ground-motion records available too if an analysis needs to reconstruct or verify inputs from them.

## Connect and use

Keep a working copy of the analysis kit on each device's internal disk. Use the local project/code checkout and local Python interpreter. Copy `config/ssd.example.json` to `config/ssd.local.json` on that device. Set:

- `project_root`: that device's RC Structure code folder.
- `python_executable`: that device's analysis-capable interpreter.
- `dataset_base`: the actual mounted `StructuralResearch/datasets` directory.
- `device_id`: a useful device name, such as laptop or desktop.

Drive letters and mount paths can differ. Resolve the currently mounted volume and confirm its contents before running preflight; do not assume a drive letter identifies the same disk. Do not guess a username, copy a Python environment between devices, or rewrite paths embedded in raw metadata. The dataset entries remain relative to `dataset_base`.

Run the TASK_PROMPT with `config/ssd.local.json` explicitly selected. Write analysis intermediates and dated outputs to the active device's local analysis workspace. After a report is complete, copy its output folder to the SSD's reports area if you want it available elsewhere. A Codex task does not automatically follow the SSD; open a local task on the device in use with the same prompt and that device's profile.

## Keep copied data identifiable

Copy a stable completed batch, or a deliberately documented partial batch whose files are no longer being written. Keep live generator state on the generating machine. Do not run generator merge/status-refresh commands merely to prepare an analysis copy.

Record source device, source root, copy date, generation/code revision if known, dataset/plan identity, and exact copied file roster. Verify source/destination file sizes and SHA256 fingerprints after copying. For very large datasets, use a per-file transfer manifest generated once for the stable batch; preflight hashes only the small files it actually reads and does not verify the entire transfer.

Treat changed plans or conflicting files as separate versions until reconciled. Never overwrite a conflicting same-named case automatically. Preserve experiment + case + run + record/scale identity so copies from two devices are not counted as independent observations. An unchanged plan hash alone does not prove response files are identical.

The SSD should not be the only copy of valuable generated data. Keep the authoritative desktop copy or a separate backup before removing data from any machine. No automatic backup/sync or recurring monitoring is configured by this kit.

## Desktop profile update

The user-provided desktop repository is recorded in DESKTOP_SETUP.md. Use `config/desktop-ssd.local.json` on that machine; it preserves the desktop repository path and leaves the SSD dataset mount and Python interpreter to be configured locally. The existing `config/ssd.local.json` remains the laptop profile.
