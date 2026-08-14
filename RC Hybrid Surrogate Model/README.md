# RC Hybrid Surrogate Model

Independent GNN-LSTM surrogate for the parameterized reinforced-concrete
OpenSees dataset. It does not use the predecessor project's static GNN model.

## Baseline task

The first model predicts the causal roof displacement history in global X and
Y from:

- the variable structural graph (`x`, `edge_index`, and `edge_attr`),
- 43 global building/design features,
- two-component ground-motion acceleration, and
- 12 record-level features (six per component).

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

```powershell
& 'C:\Users\andro\anaconda3\envs\OpPy\python.exe' 'RC Hybrid Surrogate Model\train.py'
```

The baseline configuration is in `configs/baseline.json`. Sequence stride is
four by default to make initial experiments tractable; the raw full-resolution
arrays remain unchanged. Checkpoints, normalization, split assignments, and
history are written beneath `outputs/baseline`.

Multiple merged or still-separate dataset roots can be supplied by repeating
`--dataset-root`. Duplicate case/run identities are rejected.

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
