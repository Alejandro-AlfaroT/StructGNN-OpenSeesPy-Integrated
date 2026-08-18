# RC Hybrid Surrogate Model

Independent GNN-LSTM surrogate for the parameterized reinforced-concrete
OpenSees dataset. It does not use the predecessor project's static GNN model.

## Baseline task

The first model predicts the causal roof displacement history in global X and
Y from:

- the variable structural graph (`x`, `edge_index`, and `edge_attr`),
- 43 global building/design features,
- two-component ground-motion acceleration, and
- 12 record-level features (six per component), and
- 52 derived ground-motion, modal, and structure-spectrum interaction features.

The GNN produces one structural embedding. That embedding conditions a
unidirectional LSTM at every timestep. Resultant roof displacement is derived
from predicted X/Y instead of learned as a redundant third channel.

## Leakage control

Train, validation, and test splits are grouped by the ordered X/Y ground-motion
record pair. A record pair can never appear in more than one split. All
normalization statistics are calculated from the training split only.

## Smoke test

From the repository root:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' 'RC Hybrid Surrogate Model\train.py' --smoke-test
```

This loads two real samples, batches their graphs/sequences, performs a forward
and backward pass, and exits without training.

## Training

Build or refresh the derived feature cache after importing additional cases:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' `
  'RC Hybrid Surrogate Model\build_engineered_features.py'
```

This reads existing acceleration histories and modal diagnostics without
modifying the generated OpenSees dataset or rerunning an NTHA. Ground-motion
features are computed once per unique record pair and reused across structures.

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' 'RC Hybrid Surrogate Model\train.py'
```

Running `train.py` without arguments uses the 10-feature physics ablation for
the 1,899-case interim dataset, with the intensity-stratified split described
below. It retains raw X/Y acceleration, four modal features, and six
structure-spectrum interaction features while excluding the 42 standalone
motion descriptors. It reads `derived_features/engineered_features.json`,
writes to `outputs/interim_1899_physics10_stratified_v1`, and enables the
auto-refreshing loss dashboard. The non-stratified version of this config is
still available at `configs/interim_1899_physics10.json` if you need to
reproduce the original split; the reusable baseline configuration remains in
`configs/baseline.json`. Sequence stride is four by default to make initial
experiments tractable; the raw full-resolution arrays remain unchanged.
Checkpoints, normalization, split assignments, and history are written beneath
the selected output directory.

For the 1,899-case interim experiment, use `configs/interim_1899.json`. It sends
approximately 82% of complete earthquake-record groups to training, keeps one
complete pair (35--36 samples) as the smallest possible leakage-free test set,
and assigns every other non-training record group to validation. An exact
ten-sample test would split a record pair and leak ground motions between
validation and test.

To generate an auto-refreshing training/validation loss dashboard, open
`outputs/interim_1899_physics10_stratified_v1/loss_curves.html` in a browser
after epoch 1. With `--live-plot`, that page refreshes every three seconds and
the training script rewrites its embedded SVG curve after every epoch:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' `
  'RC Hybrid Surrogate Model\train.py' `
  --dataset-root 'RC Structure\outputs\parameterized_2500' `
  --config 'RC Hybrid Surrogate Model\configs\interim_1899_physics10_stratified.json' `
  --output-dir 'RC Hybrid Surrogate Model\outputs\interim_1899_physics10_stratified_v1' `
  --engineered-feature-cache 'RC Hybrid Surrogate Model\derived_features\engineered_features.json' `
  --device cuda `
  --live-plot
```

In PyCharm, use `train.py` as the script, the repository root as the working
directory, the arguments after `train.py` as parameters, and the OpPy Conda
environment as the interpreter.

Multiple merged or still-separate dataset roots can be supplied by repeating
`--dataset-root`. Duplicate case/run identities are rejected.

## Split stratification

The interim dataset spans only 54 distinct ground-motion record pairs, so a
plain random group shuffle (`grouped_split`) can hand validation's ~9 groups a
subset that is, by chance, systematically more or less severe than the ~44
training groups. In the `interim_1899_physics10` split (seed `20260809`) this
happened: validation's mean intensity score (Arias intensity, PSa at 1 sec,
and PGV, X and Y, combined as an average percentile rank in `[0, 1]`) was
0.679 against train's 0.467 -- a gap large enough to explain a validation-loss
plateau that more dropout/weight_decay does not fix, since it is a split
composition issue rather than overfitting.

Set `"stratify_split": true` in a config to use `stratified_grouped_split`
instead: it keeps the same group-leakage guarantee (a whole record pair still
goes to exactly one split) but orders groups by
`features.load_group_intensity_scores(...)` and hands them out with a deficit
round-robin so every split gets a proportional spread of mild-to-severe
motions instead of a random subset. `configs/interim_1899_physics10_stratified.json`
is the physics10 config with this enabled; with the same seed it brings
validation's mean score to 0.477, in line with train's 0.514. (The single
`test` group is exempt from this balancing -- with a target of one, "balance"
isn't a meaningful concept for it, and today `train.py` never scores the test
split anyway.) `train.py` always prints each split's mean intensity score at
startup, stratified or not, as a self-check.

## Raw parameter embedding projector

Export standardized structural-only and structure-plus-record parameter spaces:

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' 'RC Hybrid Surrogate Model\export_raw_embeddings.py'
```

Each output contains `vectors.tsv` and `metadata.tsv` for loading directly into
the TensorBoard Embedding Projector. The exporter also writes raw vectors,
feature names, standardization values, and three-component PCA coordinates.
These visualization statistics use all discovered cases for exploratory data
coverage only; model-training normalization remains training-split-only.
