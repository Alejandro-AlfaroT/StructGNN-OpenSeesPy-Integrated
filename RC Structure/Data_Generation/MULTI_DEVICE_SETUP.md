# Multi-device parameterized generation

Use deterministic, non-overlapping case ranges and a local output directory on
each device. Do not let two schedulers write to the same live OneDrive, network,
or shared output folder.

## Assigned ranges

- Device 1: cases 1 through 1250
- Device 2: cases 1251 through 2500

The ranges are inclusive. `--run-limit` limits the number of unfinished cases
attempted by one invocation inside that device's assigned range.

## Prepare Device 2

1. Copy or clone the repository onto Device 2.
2. Copy the complete `RC Structure/Ground_Motions` directory.
3. Copy the authoritative `parameter_plan.json` and `parameter_plan.csv` from
   Device 1's `RC Structure/outputs/parameterized_2500` into the same relative
   output directory on Device 2.
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
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --num-cases 2500 --case-start 1251 --case-end 2500 --run-limit 100 --workers 4 --output-root 'RC Structure\outputs\parameterized_2500' --python-exe "$env:CONDA_PREFIX\python.exe"
```

## Stop safely

Run this in a second PowerShell window on the device being stopped:

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --output-root 'RC Structure\outputs\parameterized_2500' --request-stop
```

## Merge Device 2 results

After Device 2's scheduler has stopped, copy only its
`RC Structure/outputs/parameterized_2500/cases` contents into Device 1's
corresponding `cases` directory. Do not overwrite Device 1's root manifests or
state files. The assigned ranges ensure that case directories do not collide.

Then rebuild Device 1's manifests without launching analyses:

```powershell
python -B 'RC Structure\Data_Generation\Generate_Parameterized_Dataset.py' --num-cases 2500 --output-root 'RC Structure\outputs\parameterized_2500' --refresh-status
```
