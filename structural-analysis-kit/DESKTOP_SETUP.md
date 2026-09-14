# Desktop setup

User-provided repository root:

```text
C:\Users\andro\Documents\GitHub\StructGNN-OpenSeesPy-Integrated
```

The profile proposes `RC Structure` beneath that repository because this is the inspected laptop layout. The desktop path is not present on the current laptop, so the subfolder and dataset contents have not been verified. On the desktop, locate `Structure_Parameters.py` and `Data_Generation` and set `project_root` to their containing directory if the layout differs.

- Use `config/desktop.local.json` for data in the desktop project outputs.
- Use `config/desktop-ssd.local.json` for datasets on the mounted SSD.
- Set `python_executable` to the desktop interpreter.
- Set `dataset_base` to the current mounted dataset folder for SSD mode.

Keep the laptop profiles unchanged. Re-run preflight on the desktop after setting its interpreter and checking the layout.
