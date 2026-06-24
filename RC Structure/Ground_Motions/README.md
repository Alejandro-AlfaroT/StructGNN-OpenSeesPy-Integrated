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
