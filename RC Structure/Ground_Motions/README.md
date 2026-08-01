# Ground Motion Database

This folder is the local ground-motion catalog for RC Structure nonlinear
time-history analysis and future data generation.

## Folder Layout

- `metadata/record_manifest.csv`: one row per usable horizontal component.
- `metadata/record_sets.csv`: named subsets for training, validation, testing,
  or scenario studies.
- `raw/`: original downloaded files, kept out of git.
- `processed/`: normalized acceleration histories, kept out of git.
- `scaled/`: analysis-ready scaled acceleration histories, kept out of git.

## Recommended Sources

1. PEER Ground Motion Database, especially NGA-West2 for shallow crustal
   active-tectonic records. PEER downloads are unscaled and unrotated, so store
   any scale factors separately in the manifest rather than overwriting raw
   records.
2. CESMD for U.S. and worldwide strong-motion records, especially if you want
   COSMOS-format data or structural/ground response stations.
3. USGS/ANSS strong-motion products for event/station discovery and U.S. event
   context.

## Initial Selection Targets

Start with a balanced pilot set before going large:

- 20 to 40 records, two horizontal components when available.
- Moment magnitude bins: 5.5-6.0, 6.0-6.5, 6.5-7.0, 7.0+.
- Distance bins: 0-10 km, 10-30 km, 30-60 km.
- Site bins by `vs30_m_per_s`: soft/stiff soil and rock.
- Include metadata for fault type, event name, station, component angle, PGA,
  PGV, significant duration, and scale factor.

## Units

The model uses kip-inch-second units. Store raw accelerations in their source
units, but convert analysis input to `in/sec^2` before creating an OpenSees
`Path` time series. For accelerations in `g`, multiply by `386.4`.

## Record ID Convention

Use stable IDs:

```text
SOURCE_EVENT_STATION_COMPONENT
```

Example:

```text
PEER_LOMAP_GILROYARRAY1_000
```

Do not encode scale factor in the record ID. Scale factors belong in metadata.

## Imported PEER MLE Catalog

The current local catalog was imported from `MLEarthquakeRecords.zip`. From
the repository root, import or refresh it with:

```powershell
$python = Join-Path $env:USERPROFILE "anaconda3\envs\OpPy\python.exe"
& $python -B ".\RC Structure\Ground_Motions\import_peer_zip.py" `
  (Join-Path $env:USERPROFILE "Downloads\MLEarthquakeRecords.zip") --overwrite
```

The importer writes one manifest row per usable horizontal acceleration
component. Vertical components are preserved only as source metadata because the
current NTHA workflow is organized around horizontal excitation components.

Current catalog summary:

- 164 horizontal acceleration records in the catalog.
- 108 usable records and 54 usable PEER pairs after the 15,000-point filter.
- 13 unique earthquake events in the full catalog; 11 in the usable subset.
- 56 records are retained but marked unusable because their pair exceeds
  15,000 acceleration points.
- Magnitude range: 6.53 to 7.36.
- Rupture-distance range: 11.03 km to 39.91 km.
- Vs30 range: 254.26 m/s to 442.61 m/s.
- PGA range: 0.0620 g to 0.5355 g.


## Running Nonlinear Time-History Analysis

After importing the PEER records, run a single bidirectional record pair with:

```powershell
$python = Join-Path $env:USERPROFILE "anaconda3\envs\OpPy\python.exe"
& $python -B ".\RC Structure\Ground_Motion_Main.py" --result-id 1 --catalog-summary
```

Run the first three record pairs in the manifest-backed set:

```powershell
& $python -B ".\RC Structure\Ground_Motion_Main.py" --limit 3
```

Run X-only excitation for a specific component:

```powershell
& $python -B ".\RC Structure\Ground_Motion_Main.py" `
  --record-id-x RSN15_KERN_TAF021 --x-only
```

Outputs are written under `outputs/ntha/`. Each run stores status, record
summaries, gravity and modal results, time history, story-drift peaks, node
displacement envelopes, and element end-force envelopes.

Scale factors are applied in memory during OpenSees analysis. The canonical
files under `processed/` remain unscaled and must not be edited per run.
