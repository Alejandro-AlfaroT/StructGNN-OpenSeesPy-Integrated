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

Expected: `Plan SHA256: 0ECF5651E131F8A1A7F1F58F4637E3222F411AEE89407D2C56926370F9F1B890`

Mismatch = stop. Causes in order: `git pull` not run, stray edit in `RANGES`
or `SEISMIC_SITES`, wrong `--count`. The launcher also refuses to add cases to
an output root whose `plan.json` carries a different SHA.

## 3. Case ranges

| machine | case-start | case-end | serial-equivalent | wall at 4 workers |
|---|---|---|---|---|
| 1 | 1 | 25 | see launch message | |
| 2 | 26 | 50 | | |
| 3 | 51 | 75 | | |
| 4 | 76 | 100 | | |
| 5 | 101 | 125 | | |
| 6 | 126 | 150 | | |

Case size varies 10x (3x2x4 to 6x6x9); the two 6x6x9 cases are `case_0015`
(machine 1) and `case_0058` (machine 3). A machine is not stuck because one
case has run for an hour.

## 4. Launch

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --count 150 --case-start 1 --case-end 25 --workers 4 --probe-assertions --output-root "<local>\dv150"
```

Change only `--case-start` / `--case-end`. Everything else identical on every
machine. Leave the terminal open; the launcher writes `<local>\dv150\run_log.txt`
and `<local>\dv150\case_XXXX\{design.json, result.json, log.txt, stderr.txt}`.

Before walking away: the process dies with the Windows session, so no
sign-out, and sleep must be off (`powercfg /change standby-timeout-ac 0`, or
Settings > Power). Lock the screen instead of signing out.

## 5. Monitoring

```
& "<python>" -c "import json,glob,collections; rs=[json.load(open(p)) for p in glob.glob(r'<local>\dv150\case_*\result.json')]; c=collections.Counter(r['status'] for r in rs); print(dict(c), 'accepted', sum(1 for r in rs if r.get('accepted')), 'GB %.1f' % (sum(r.get('design_json_bytes',0) for r in rs)/1e9)); [print(r['case']['case_id'], r['status'], str(r.get('error',''))[:120]) for r in rs if r['status']!='designed']"
```

or just `Get-Content "<local>\dv150\run_log.txt" -Tail 8`.

Normal: four `python` processes at 0.7–2 GB each; one line per finished case
in `run_log.txt`; per-case time from 1.5 min (small 4-story) to about an hour
(6x6x9); `design.json` 60–500 MB.

Not normal: `status: error` in a `result.json` — read its `error` and
`traceback` and the case's `stderr.txt`. `accepted=False` with `fail>0` is a
result, not a malfunction — that case is telling you something about the
methodology; leave it and read `summary.md` at the end.

Interrupted (power, killed terminal, reboot): re-run the same launch line.
Designed cases are skipped, an interrupted case is retried from scratch (the
launcher clears its lock). Nothing to clean up by hand.

Graceful stop (in-flight cases finish, no new ones start; relaunching resumes):

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --request-stop --output-root "<local>\dv150"
```

## 6. Collect and merge

From each machine, copy the case directories to one place (about 25 GB total):

```
robocopy "<local>\dv150" "E:\StructGNN_outputs\dv150" /E /MT:16 /R:2 /W:2 /XF summary.* run_log.txt
```

`plan.json` is the same on every machine (same SHA), so copying it over itself
is fine. Then, on the machine holding the merged root:

```
& "<python>" -B "<repo>\RC Structure\Design\Verify_Designs.py" --count 150 --probe-assertions --summarize-only --output-root "E:\StructGNN_outputs\dv150"
```

Read `summary.md` top to bottom:

1. `150 designed, N accepted, 0 errors, 0 not yet run` — anything not yet run
   means a machine's copy is incomplete.
2. `Request identities among designed cases: 1` — more than one means a
   machine designed with different code or config; its cases are not the same
   experiment. Find it in the `host` column.
3. Failed-check histogram — which qualification items fail and how often.
4. Open-item histogram — should be empty under PROBE assertions.
5. The table: section variety, T1 range, DCR near the 0.60–0.95 band, hoops.

Then pick 3–5 cases across sites and heights for SAP2000
(`Design/Export_SAP2000.py <case dir>` → import → `Design/Compare_SAP2000.py`).

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
