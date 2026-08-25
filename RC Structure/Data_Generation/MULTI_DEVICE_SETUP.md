# Multi-device parameterized generation

Use deterministic, non-overlapping case ranges and a local output directory on
each device. Do not let two schedulers write to the same live OneDrive, network,
or shared output folder.

## Assigned ranges

The 2,500-case plan was generated across three devices:

- Device 1: cases 1 through 1250
- Device 2: cases 1251 through 1899
- Device 3: cases 1900 through 2500

The ranges are inclusive and must not overlap. `--run-limit` limits the number
of unfinished cases attempted by one invocation inside that device's assigned
range.

Any partition of 1–2500 works as long as the ranges are contiguous and
disjoint; size each device's share to its throughput. Cases are dispatched to
workers in ascending order within a device, so a device's range fills in
roughly ascending order even though individual completion order varies with
per-case runtime. A device that finishes its whole range therefore yields a
complete contiguous block, which can be used for training before the slowest
device is done. That is how the 1,899-case interim dataset arose: devices 1
and 2 had finished cases 1–1899 while device 3 was still working through
1900–2500.

At startup, the scheduler prints the SHA256 of `parameter_plan.csv`. Confirm
that this value matches across every device before allowing generation to
continue.

## Prepare a secondary device

Repeat these steps on every device other than Device 1.

1. Copy or clone the repository onto the device.
2. Copy the complete `RC Structure/Ground_Motions` directory.
3. Copy the authoritative `parameter_plan.json` and `parameter_plan.csv` from
   Device 1's `RC Structure/outputs/parameterized_2500` into the same relative
   output directory on that device.
4. Create the generation environment from the repository root:

   ```powershell
   conda env create -f 'RC Structure\environment-generation.yml'
   conda activate OpPy
   ```

5. Verify the environment:

   ```powershell
   python -c "import numpy; import openseespy.opensees; print('generation environment OK')"
   ```

## Run bounded batches

Device 1:

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --num-cases 2500 --case-start 1 --case-end 1250 --run-limit 100 --workers 9 --output-root 'RC Structure\outputs\parameterized_2500' --python-exe "$env:CONDA_PREFIX\python.exe"
```

Device 2 (choose a worker count appropriate for that CPU):

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --num-cases 2500 --case-start 1251 --case-end 1899 --run-limit 100 --workers 4 --output-root 'RC Structure\outputs\parameterized_2500' --python-exe "$env:CONDA_PREFIX\python.exe"
```

Device 3 (choose a worker count appropriate for that CPU):

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --num-cases 2500 --case-start 1900 --case-end 2500 --run-limit 100 --workers 4 --output-root 'RC Structure\outputs\parameterized_2500' --python-exe "$env:CONDA_PREFIX\python.exe"
```

`--num-cases 2500` stays the same on every device; it defines the plan, not the
device's share. Only `--case-start` and `--case-end` differ.

## Stop safely

Run this in a second PowerShell window on the device being stopped:

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --output-root 'RC Structure\outputs\parameterized_2500' --request-stop
```

## Merge secondary device results

Merge each secondary device separately, in any order. After that device's
scheduler has stopped, copy only its
`RC Structure/outputs/parameterized_2500/cases` contents into Device 1's
corresponding `cases` directory. Do not overwrite Device 1's root manifests or
state files. The assigned ranges ensure that case directories do not collide,
so Device 2's and Device 3's case folders merge into Device 1 without
overlapping each other.

Then rebuild Device 1's manifests without launching analyses:

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --num-cases 2500 --output-root 'RC Structure\outputs\parameterized_2500' --refresh-status
```

## Cases 2501–3800 ground-motion expansion

The expansion plan is stored separately at
`RC Structure/outputs/parameterized_expansion_2501_3800`. It contains 1,300
new geometries, uses case IDs 2501–3800, and assigns each of the 26 additional
PEER pairs to exactly 50 structures. The record set has a 30,000-point ceiling;
the two 60,000-point pairs remain excluded.

Run the one-case-per-record pilot first:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' -B `
  'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' `
  --num-cases 1300 --run-limit 26 --workers 4 `
  --set-name peer_expansion_30k --max-npts 30000 `
  --geometry-offset 2500 --case-id-offset 2500 `
  --output-root 'RC Structure\outputs\parameterized_expansion_2501_3800' `
  --python-exe 'C:\Users\andro\anaconda3\envs\OpPy\python.exe'
```

After validating the pilot, run the same command without `--run-limit 26` to
resume and complete the plan. The scheduler skips completed cases. For this
plan, `--case-start` and `--case-end` refer to positions 1–1300: position 1 is
`case_2501` and position 1300 is `case_3800`.

The expansion splits across devices the same way, using positions rather than
case IDs. For example, a three-way split is `--case-start 1 --case-end 650`,
`--case-start 651 --case-end 1000`, and `--case-start 1001 --case-end 1300`.
Every device must pass the same `--num-cases 1300`, `--geometry-offset 2500`,
`--case-id-offset 2500`, `--set-name`, and `--max-npts` values, and must use
the `parameterized_expansion_2501_3800` output root. Merge the results into the
primary device exactly as described above, then rebuild manifests with
`--refresh-status` against that same output root.

Stop the expansion from a second PowerShell window with:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' -B `
  'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' `
  --output-root 'RC Structure\outputs\parameterized_expansion_2501_3800' `
  --request-stop
```
