# The hypergraph stage

## Why a hypergraph and not a graph

A graph edge relates exactly two nodes. Many of the relations that matter here
are relations over **sets**:

- "the twelve studies whose fused representation is closest to this one"
- "every AP film in the cohort"
- "the cluster of training cases that present cardiomegaly this way"

Encoding "these thirteen studies belong together" as 78 pairwise edges loses the
thing you were trying to say. A hyperedge says it directly, and the HGNN
normalisation `De^-1` then divides a hyperedge's message by its size, so a large
attribute group does not drown out a tight neighbourhood.

## The operator

Feng et al. (2019), *Hypergraph Neural Networks*:

```
X' = σ( Dv^-1/2 · H · W · De^-1 · Hᵀ · Dv^-1/2 · X · Θ )
```

| symbol | shape | meaning |
|---|---|---|
| `H` | `[N, E]` | incidence, `H[i,e] = 1` when study *i* is in hyperedge *e* |
| `W` | `[E, E]` | diagonal, learned hyperedge weights |
| `De` | `[E, E]` | diagonal edge degrees, `|e|` |
| `Dv` | `[N, N]` | diagonal node degrees, `Σ_{e ∋ i} w_e` |
| `Θ` | `[d, d']` | the layer's linear map |

Computed in `hgnn.py` as two sparse mat-muls — never materialising `N × N`:

```python
inv_sqrt  = (H @ w).clamp_min(1e-6).pow(-0.5)      # [N, 1]
scattered = (Hᵀ @ (x * inv_sqrt)) * (w / edge_sizes)   # [E, d]
out       = (H @ scattered) * inv_sqrt                  # [N, d]
```

At N = 12k with ~40k hyperedges and ~450k non-zeros, one layer is a few
milliseconds and the whole graph is a couple of hundred megabytes.

## The six hyperedge families

| group | one hyperedge is… | built from | count |
|---|---|---|---|
| `knn_fused` | `{i} ∪ kNN(i)` in fused space | features | N |
| `knn_image` | `{i} ∪ kNN(i)` in image space | features | N |
| `knn_text` | `{i} ∪ kNN(i)` in report space | features | N |
| `meta_view` | studies sharing a view (PA/AP/…) | metadata | small |
| `meta_demo` | studies sharing (sex, age decade) | metadata, when available | small |
| `proto_label` | a k-means prototype of training positives | **train labels** | ~24 |

Three similarity views rather than one, because a case can be visually typical
and textually unusual, or the reverse — and the learned per-group weight then
tells you which mattered. Those weights are reported in `metrics.json` and
plotted in `figures/hyperedge_weights.png`; they are a result in their own right.

**Attribute groups are size-capped** (`hypergraph.max_edge_size`, default 256).
A hyperedge containing half the dataset is a bias term wearing a relation's
clothes; oversized groups are split into bounded random sub-edges instead.

## Leakage rules

1. Labels are read **only** for rows whose split is `train`.
2. `proto_label` centroids are fitted on training positives alone. Training nodes
   join their own cluster; **every other node joins by cosine distance to the
   centroid**, never by its own label.
3. Validation and test nodes *do* participate in message passing. That uses their
   **features**, not their labels, and is the standard transductive HGNN setting.
   `hypergraph.transductive: false` builds the graph from training nodes only,
   for a strictly inductive comparison — run both and report both.

## The read-out is a residual

```python
final_logits = backbone_logits + delta_scale * head([x ; HGNN(x)])
```

with the head's final layer zero-initialised, so training **starts exactly at the
backbone's behaviour**.

This is not a stylistic choice. Measured on this codebase with an otherwise
identical setup, a head trained from scratch on the graph reached **0.639** macro
AUROC where the backbone it sat on top of reached **0.734**. The HGNN sees only a
few thousand training nodes; relearning the whole decision from that is strictly
harder than learning a correction. As a residual, the same configuration reached
**0.742**.

It also makes the ablation honest: `fusion` and `fusion_hypergraph` now differ by
exactly the graph's contribution, not by "one head had more training signal than
the other".

`metrics.json` records `mean_logit_correction` — the average absolute size of
that correction. If it is near zero, the graph found nothing to add, and that is
worth saying plainly.

## Inference on one unseen patient

An HGNN reasons over all nodes at once; a deployed model gets one study. The
protocol here is the **frozen-bank extension**:

> The query node joins new hyperedges of its own — its kNN edges against the
> exported bank, and its nearest prototype edge. The bank's own edges, sizes and
> node degrees are left untouched.

Three properties follow, all of them things you want in a served model:

1. **Stability.** Nothing about the bank changes, so two people uploading
   simultaneously cannot influence each other's result. Predictions are
   reproducible forever.
2. **Cost.** `O(k · d)` per layer instead of a full re-propagation. Measured
   ~4 ms of the ~95 ms end-to-end request.
3. **Exactness under the protocol.** The bank's per-layer hidden states are
   exported, so the query's layer-2 aggregation uses its neighbours' *true*
   layer-1 outputs. `forward_query` computes the frozen-bank quantity exactly —
   it is not an approximation of it.

What it is **not**: identical to rebuilding the whole hypergraph with the query
inserted. Inserting a node would change its neighbours' degrees and would let it
appear in other nodes' kNN edges. That difference is measured, not assumed —
`dvlhg eval` runs both paths on held-out studies and reports:

```
inductive vs transductive on 64 held-out studies:
  mean |dp| = 0.0000, AUROC 0.7300 vs 0.7309
```

If that gap is ever large, the reported numbers do not describe what the API
does, and the report says so before anyone quotes them.

### What the bank contains

`export/bank.npz` holds, for **training and validation nodes only**:

- fused / image / text feature vectors
- per-layer HGNN hidden states
- node degrees
- view strings, and the prototype centroids

`export/bank_meta.csv` holds study ids, splits and labels for the "nearest cases"
panel in the UI.

No image and no report text is ever exported. Test films are excluded entirely,
which also means a query can never retrieve itself.

## Hyperparameters worth sweeping

| knob | default | effect |
|---|---|---|
| `knn_fused` / `knn_image` / `knn_text` | 10 / 8 / 8 | neighbourhood size; 0 disables that family |
| `layers` | 2 | 3+ over-smooths quickly on kNN graphs |
| `edge_dropout` | 0.1 | DropEdge; raise if the HGNN overfits |
| `dropout` | 0.3 | feature dropout in the encoder |
| `max_edge_size` | 256 | cap on attribute-group hyperedges |
| `proto_clusters` | 24 | total prototype edges across the four findings |
| `transductive` | true | false = train-only graph |

Because stage 3 trains in seconds, sweep these properly rather than guessing.
