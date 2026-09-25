# 150-Case Design Verification — Operating Checklist

150 plan cases (the first 150 of the generation plan: same seed, shuffle and
hazard round-robin as `build_plan`), **design only**. No ground motions, no
NTHA. Each case runs the full SMRF loop (drift screen, capacity design, slab
transfer and reinforcement, qualification) under PROBE assertions and keeps its
`design.json`. Six machines, disjoint case ranges, merged afterwards.

What it produces:

- `summary.md` / `summary.csv` — accepted / fail / open per case, failed-check
  and open-item histograms, sections, bars, hoops, T1, DCR, which host designed it.
- 150 `design.json` artifacts for the SAP2000 spot checks
  (`Design/Export_SAP2000.py <case dir>`) that back the `IndependentVerification`
  assertions in `Design/Config.py`.

What it does not do: certify anything. `--probe-assertions` fills the three
assertion blocks with labelled PROBE values so the whole design path runs
(without them slab reinforcement is skipped and SCWB stays `not_evaluated`).
The real assertions are yours to write after the SAP comparison; the NTHA run
then redesigns every case under them (the request identity differs, so PROBE
designs are never reused by the dataset run).

## 1. Per-machine setup

```
git pull
git rev-parse HEAD
```

HEAD must be the commit in the launch message. Ground motions are **not**
needed for this run.

## 2. Plan SHA — don't skip

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --count 150 --plan-only
```

Expected: `Plan SHA256: CD813D4D66A4958B51069F67F9010EE243E6544D02E7ACB51E916EEA7653B94E`
(plan of 2026-09-24, story heights 12-16 ft; the dv150 roots v7-v10 were
designed under the 10-14 ft plan `0ECF5651...` and cannot be resumed or
extended under this one -- start a new root)

Mismatch = stop. Causes in order: `git pull` not run, stray edit in `RANGES`
or `SEISMIC_SITES`, wrong `--count`. The launcher also refuses to add cases to
an output root whose `plan.json` carries a different SHA.

## 3. Case ranges

| machine | case-start | case-end | serial-equivalent | wall at 4 workers | biggest case |
|---|---|---|---|---|---|
| 1 | 1 | 25 | 2.5 h | ~0.6 h | case_0015 6x6x9, ~30 min, ~380 MB |
| 2 | 26 | 50 | 4.5 h | ~1.1 h | case_0041 5x6x9 |
| 3 | 51 | 75 | 3.4 h | ~0.9 h | case_0058 6x6x9 |
| 4 | 76 | 100 | 2.4 h | ~0.6 h | case_0096 6x4x9 |
| 5 | 101 | 125 | 2.6 h | ~0.7 h | case_0123 4x6x9 |
| 6 | 126 | 150 | 2.5 h | ~0.6 h | case_0145 5x6x9 |

Estimates from three measured cases (3x4x4 84 s / 60 MB, 6x5x4 519 s /
154 MB, 6x6x9 1039 s / 381 MB, peak 1.2 GB RAM per worker); per-iteration
cost grows about as members^1.6 and the loop takes 2-4 iterations. Expect
1.5-2x these on the lab machines. Whole plan: ~18 h serial, ~21 GB.
Case size varies 10x; a machine is not stuck because one case has run for
half an hour.

## 4. Launch

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --count 150 --case-start 1 --case-end 25 --workers 4 --probe-assertions --probe-date 2026-09-16 --output-root "<local>\dv150"
```

Change only `--case-start` / `--case-end`. Everything else identical on every
machine. Leave the terminal open; the launcher writes `<local>\dv150\run_log.txt`
and `<local>\dv150\case_XXXX\{design.json, result.json, log.txt, stderr.txt}`.

Choose one shared `--probe-date` for the experiment (the date above is an
example). It is required when creating a new PROBE root and must match on
every device. An existing root retains its recorded date; it must not be
silently changed on a later launch. Plan SHA covers geometry/hazard only,
not the assertion date or source/config identity.

Before walking away: the process dies with the Windows session, so no
sign-out, and sleep must be off (`powercfg /change standby-timeout-ac 0`, or
Settings > Power). Lock the screen instead of signing out.

## 5. Monitoring

```
& "<python>" -c "import json,glob,collections; rs=[json.load(open(p)) for p in glob.glob(r'<local>\dv150\case_*\result.json')]; c=collections.Counter(r['status'] for r in rs); print(dict(c), 'accepted', sum(1 for r in rs if r.get('accepted')), 'GB %.1f' % (sum(r.get('design_json_bytes',0) for r in rs)/1e9)); [print(r['case']['case_id'], r['status'], str(r.get('error',''))[:120]) for r in rs if r['status']!='designed']"
```

or just `Get-Content "<local>\dv150\run_log.txt" -Tail 8`.

Normal: four `python` processes at up to ~1.2 GB each; one line per finished
case in `run_log.txt`; per-case time from 1.5 min (small 4-story) to about
half an hour (6x6x9); `design.json` 30–400 MB.

Not normal: `status: error` in a `result.json` — read its `error` and
`traceback` and the case's `stderr.txt`. `accepted=False` with `fail>0` is a
result, not a malfunction — that case is telling you something about the
methodology; leave it and read `summary.md` at the end.

Interrupted (power, killed terminal, reboot): re-run the same launch line.
Completed designs are reused only after checking the actual artifact against
the current geometry, hazard, code, configuration and PROBE date, then
recomputing qualification. New results also bind the artifact's SHA256.
Old-code or tampered artifacts are reported as errors, not overwritten; use
a new output root for the changed methodology and preserve the old evidence.

Local OS leases exclude simultaneous launchers and surviving workers. Never
share a live output root over a network or sync tool. A leftover
`.design.json.lock` is **not automatically deleted**: establish that its owner
has exited before manual recovery. Partial design files are preserved too.
Persistent `.launcher.lease` / `.worker.lease` files are normal; their presence
does not mean they are held, and they should not be deleted.

Graceful stop (in-flight cases finish, no new ones start; relaunching resumes):

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --request-stop --output-root "<local>\dv150"
```

## 6. Collect and merge

From each machine, copy the case directories to one place (about 25 GB total):

```
robocopy "<local>\dv150" "E:\StructGNN_outputs\dv150" /E /MT:16 /R:2 /W:2 /XF summary.* run_log.txt plan.json *.lease
```

Copy only from stopped runs and merge disjoint case folders without
overwriting another device's cases. Plans share a geometry digest but contain
different host/launch histories: do not overwrite them. Keep a copy of each
device's plan outside the merged root for provenance. Then create the merged
root's own plan using the same count, seed, geometry offset and PROBE date:

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --count 150 --probe-assertions --probe-date 2026-09-16 --summarize-only --output-root "E:\StructGNN_outputs\dv150"
```

Read `summary.md` top to bottom:

1. `150 designed, N accepted, 0 errors, 0 not yet run` — anything not yet run
   means a machine's copy is incomplete.
2. `Methodology identities ... : 1` — more than one means a
   machine designed with different code or config; its cases are not the same
   experiment. Find it in the `host` column.
3. Failed-check histogram — which qualification items fail and how often.
4. Open-item histogram — should be empty under PROBE assertions.
5. The table: section variety, T1 range, DCR near the 0.60–0.95 band, hoops.

Then pick 3–5 cases across sites and heights for SAP2000
(`Design/Export_SAP2000.py <case dir>` → import → `Design/Compare_SAP2000.py`).

The SAP exporter is not yet validated by a live SAP import/analysis. Inspect
the imported point-moment directions, per-edge slab mesh, member weight
modifiers and every listed strength combination; select those combinations
for concrete design explicitly. SAP's beam steel area is not a utilization
ratio. Missing comparison tables/ratios produce `incomplete` (exit code 2),
not a pass. P-Delta and spatial mass mapping remain comparison limitations.
A numerical pass alone does not establish the engineering assertions.

## Current configuration

| setting | value | why |
|---|---|---|
| plan | first 150 of `build_plan`, seed 20260731 | the same buildings the dataset run designs first |
| hazard | round-robin over the five `SEISMIC_SITES` | 30 cases per site |
| floors | 4–9 | 2–3 story frames dropped from `RANGES` |
| assertions | PROBE (labelled) | exercises the full design path; certifies nothing |
| DCR band | 0.60–0.95 | `DESIGN_DCR_MAX` |
| workers | 4 | one fresh interpreter per case |
| `GENERATION_RELEASE_READY` | False | NTHA stays blocked until the assertions are real |

##
