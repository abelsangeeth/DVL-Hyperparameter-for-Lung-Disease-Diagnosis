# Project plan — Diffusion-Augmented Vision-Language Hypergraph Network

Predict four findings — **Atelectasis, Cardiomegaly, Edema, Pleural Effusion** —
from a chest radiograph plus its radiology report, using a diffusion model for
augmentation, a biomedical vision-language model for encoding, cross-modal
fusion, and a hypergraph neural network for relational reasoning.

This document is the plan: what to build, in what order, what will go wrong, and
how to tell whether it worked. The code that implements it is in `src/dvlhg/`,
the Colab notebooks in `notebooks/`.

---

## 1. The four decisions that determine whether this project succeeds

Most of the risk in this project is not in the architecture. It is in four
choices that are easy to get wrong and hard to notice afterwards.

### 1.1 Data access — start the paperwork today

MIMIC-CXR-JPG is credentialed. You need to complete the CITI "Data or Specimens
Only Research" course, upload the certificate to PhysioNet, and sign the
MIMIC-CXR-JPG data use agreement. **This takes days, sometimes a week or two.**

Start it before you write any code. While you wait, the whole pipeline runs on
**Open-i / Indiana University CXR** (`--set data.source=openi`), which is public,
needs no account, and has real films with real reports. Everything downstream is
unchanged; only the numbers differ.

### 1.2 Text leakage — the single biggest trap

MIMIC-CXR's labels were produced by the **CheXpert labeler, an NLP tool that
read FINDINGS and IMPRESSION**. If you feed those sections to the model and
report the AUC, you have built a system that re-reads the sentence the label came
from. You will see ~0.97–0.99 AUROC. It means nothing, and the first person who
asks "where do the labels come from?" will say so.

This project defaults to `text.mode: indication`: the **pre-read** sections only
— reason for exam, clinical history, technique, comparison. That is genuine
context a clinician has *before* the film is read, and it makes the text channel
informative without being circular.

The leaky modes still exist, clearly named, so you can **quantify the gap** — a
far more interesting result than either number alone. See `docs/LEAKAGE.md`.

### 1.3 Diffusion scope — be honest about what fits

A 128 px, ~22M-parameter conditional UNet trained for 30 epochs on ~10k films
finishes on a free T4 in under an hour. It will not produce diagnostic-quality
radiographs, and claiming otherwise invites a question you cannot answer.

Its samples are therefore used as **extra training rows at a reduced loss
weight**, never as a replacement for real data. Whether they help is settled by
the ablation, not by assertion.

### 1.4 What "hypergraph inference" means for one new patient

An HGNN is transductive: it reasons over all nodes at once. A deployed system
gets one study at a time. The protocol here — the **frozen-bank extension** — is
that a query joins new hyperedges of its own while the bank's edges, sizes and
degrees stay fixed. That makes inference exact, cheap, and stable: one patient's
prediction can never shift another's. `dvlhg eval` measures the gap between this
and the transductive computation and prints it in the report. See
`docs/HYPERGRAPH.md`.

---

## 2. Architecture

```
                  ┌──────────────────────────────────────────┐
  chest film ────►│ BiomedCLIP ViT-B/16   (last 4 blocks     │──► 197 patch tokens
                  │                        fine-tuned)       │     + pooled embedding
                  └──────────────────────────────────────────┘
                                                                  ├─► bidirectional
                  ┌──────────────────────────────────────────┐    │   cross-attention
  report    ────► │ PubMedBERT            (last 2 blocks      │──► │   (2 blocks)
  (indication)    │                        fine-tuned)        │    │        │
                  └──────────────────────────────────────────┘    │   gated pooling
                                                                  │        ▼
  diffusion ─────► synthetic films + SDEdit refinement ───────────┘   node feature (384-d)
  (stage 1)                                                               │
                                                                          ▼
                       ┌────────────────────────────────────────────────────────┐
                       │ hypergraph over ALL studies                            │
                       │   knn_fused · knn_image · knn_text                     │
                       │   meta_view · meta_demo · proto_label                  │
                       │   X' = σ(Dv^-½ H W De^-1 Hᵀ Dv^-½ X Θ),  2 layers      │
                       └────────────────────────────────────────────────────────┘
                                                                          │
                                        final = backbone logits + hypergraph correction
                                                                          ▼
                                            4 probabilities · thresholds · calibration
                                                                          ▼
                                            FastAPI + web UI (saliency, neighbours)
```

Two design choices worth defending in a viva:

**The hypergraph predicts a correction, not a fresh decision.**
`final = backbone_logits + head([x ; HGNN(x)])`, with the head's last layer
zero-initialised. Training therefore *starts* at the backbone's behaviour. This
is not cosmetic: measured on this codebase, a head trained from scratch on a few
thousand nodes landed **below** the backbone it was meant to improve (0.639 vs
0.734 macro AUROC); as a residual it landed above it. It also makes the
"fusion vs fusion+hypergraph" ablation a clean measurement of what the graph
adds.

**Three stages, trained separately.** The hypergraph is transductive — it needs
every node's feature at once, which cannot be held on a T4 while gradients flow
through two transformers. Trained on frozen features, an HGNN epoch is
milliseconds, so the graph design can be ablated dozens of times in the time one
end-to-end epoch would take.

---

## 3. Phase plan

| Phase | What you do | Wall-clock | Blocking? |
|---|---|---|---|
| **0** | Run `notebooks/00_quickstart_smoke.ipynb` end to end on synthetic data | 5 min | no |
| **0b** | Apply for PhysioNet credentialed access | days–weeks | start now |
| **1** | Build the manifest + image cache (`01_data_and_manifest`) | 30–60 min | needs data |
| **1b** | `dvlhg check-vlm` — confirm BiomedCLIP actually loaded | 2 min | no |
| **2** | Train the diffusion model, inspect samples, generate the synthetic set (`02_diffusion`) | 45–70 min | no |
| **3** | Train the vision-language backbone (`03_backbone`) | 2–3 h | no |
| **4** | Hypergraph + evaluation + ablation + export (`04_hypergraph_eval_export`) | 20 min | no |
| **5** | Demo, `verify-serving`, write-up (`05_demo_serve`) | 1–2 h | no |
| **6** | Ablations and sensitivity runs (below) | 3–6 h | no |

Total GPU time on a free Colab T4 at the default settings: roughly **4–6 hours**,
spread over sessions. Every stage checkpoints every epoch, so a disconnect costs
at most one epoch — provided `paths.root` is on Google Drive.

### Phase 6 — the runs that make it a project rather than a demo

Each is one command; each answers a question a reader will ask.

```bash
# Does the text channel leak? (expect a large, embarrassing gap — that is the point)
dvlhg eval --set text.mode=findings         # leaky upper bound
dvlhg eval --set text.mode=findings_masked  # redacted
dvlhg eval --set text.mode=indication       # the honest number

# Does the diffusion augmentation earn its place?
dvlhg train --set train.use_synthetic=false && dvlhg eval --set train.use_synthetic=false

# Is it the hypergraph, or just more parameters?
dvlhg eval --set hypergraph.knn_fused=0 --set hypergraph.knn_image=0 --set hypergraph.knn_text=0
dvlhg eval --set hypergraph.use_proto_label=false
dvlhg eval --set hypergraph.layers=1     # and =3

# Transductive vs strictly inductive graph
dvlhg eval --set hypergraph.transductive=false

# Seed sensitivity — run three seeds before believing any gap under 0.01 AUROC
for s in 1337 7 2024; do dvlhg eval --set seed=$s; done
```

---

## 4. Evaluation protocol

Fix this before you look at a single test number.

- **Splits.** MIMIC's official patient-level split, honoured exactly. `dvlhg prep`
  refuses to continue if any patient appears in two splits.
- **Uncertain labels.** Excluded from the loss *and* from every metric
  (`data.uncertain_policy: ignore`). Per-finding `n` columns report how many
  cells actually counted. Alternatives (`ones`/`zeros`/`per_class`) are available
  and must be stated if used.
- **Thresholds and temperature** are fitted on **validation only**, then applied
  unchanged to test.
- **Test is touched once**, at the end. Treat every re-run against test as what it
  is — a hyperparameter decision made on test.
- **Metrics.** Per finding: AUROC, AUPRC, accuracy, precision, recall,
  specificity, F1. Across findings: macro averages, exact-match (subset)
  accuracy, Hamming accuracy, ECE.
- **Confidence intervals.** 1000-resample bootstrap that resamples **patients**,
  not studies — repeated films from one patient would otherwise make the interval
  look tighter than it is.
- **Reporting rule.** If the hypergraph's gain is smaller than the width of its
  confidence interval, say so. A well-reported null result is worth more than an
  overstated gain, and it is much harder to attack.

The metric implementations in `src/dvlhg/eval/metrics.py` are verified against
scikit-learn to 1e-9, including tie handling and masked cells
(`tests/test_metrics.py`).

---

## 5. Risk register

| Risk | Signal | What to do |
|---|---|---|
| PhysioNet access denied or slow | no credentials after a week | run everything on `data.source=openi`; the pipeline is identical |
| BiomedCLIP downloads config but not weights | `dvlhg check-vlm` shows ~0.50 AUROC | re-run the install cell; check network; `vlm.backbone=hf` falls back to timm ViT + Bio_ClinicalBERT |
| CUDA OOM on a T4 | crash during stage 3 | `train.batch_size=16`, `train.accum_steps=2`, or `vlm.unfreeze_vision_blocks=2` |
| Colab disconnects mid-training | session dies at ~90 min | already handled — re-run the cell, it resumes; requires `paths.root` on Drive |
| Diffusion samples are noise | sample grid looks like static | train longer, or drop `diffusion.image_size` to 64; check the loss is falling |
| Diffusion samples all identical | no diversity within a row | `diffusion.guidance_scale` too high — lower toward 1.5 |
| Suspiciously high AUC (>0.95) | every finding near-perfect | almost certainly text leakage — check `text.mode`, and read `docs/LEAKAGE.md` |
| Hypergraph makes things worse | `fusion_hypergraph` < `fusion` | already mitigated by the residual design; if it persists, raise `hypergraph.dropout`, lower `hgnn_train.lr`, or report the null result |
| Demo disagrees with the report | different numbers in the UI | run `dvlhg verify-serving` — it diffs the API against the reported predictions and names the likely cause |
| Downloading 12k JPEGs is slow | prep takes hours | it is resumable; lower `data.subset_size` to 6000 for a first pass |

---

## 6. Deliverables

1. **Code** — `src/dvlhg/`, one CLI, every stage reproducible from a config file.
2. **Notebooks** — six, numbered, each runnable top to bottom on Colab.
3. **`runs/.../reports/report.md`** — auto-generated: ablation table, per-finding
   metrics with CIs, the served-vs-evaluated check, and the caveats.
4. **Figures** — ROC, precision–recall, ablation with CIs, learned hyperedge
   group weights, training curves, diffusion sample grid.
5. **Serving bundle + web app** — drag a film in, get four calibrated
   probabilities, a saliency overlay, the report tokens the image attended to,
   and the nearest cases in the hypergraph.
6. **Written report** — structure below.

### Suggested report structure

1. **Introduction** — the four findings, why multi-label and not multi-class.
2. **Related work** — CheXpert/MIMIC baselines, BiomedCLIP/ConVIRT/GLoRIA,
   HGNN (Feng et al. 2019), medical diffusion (RoentGen).
3. **Data** — MIMIC-CXR-JPG, the subset, the uncertainty policy, **and the
   label-provenance problem in §1.2**. This section is where you earn credibility.
4. **Method** — the three stages; the hypergraph families table; the residual
   read-out; the frozen-bank inference protocol.
5. **Experiments** — main table, ablations, seed sensitivity.
6. **Results & discussion** — including anything that did not work.
7. **Limitations** — NLP-derived labels, single centre, four findings out of
   many, saliency is not evidence of validity, no prospective validation.
8. **Deployment** — the API, the `verify-serving` result, the model card.

### Questions you should be able to answer cold

- *Where do your labels come from, and what is their ceiling?* An NLP labeler
  reading reports — so the ceiling is the labeler's agreement with radiologists,
  not perfect truth.
- *Why a hypergraph instead of a graph?* A hyperedge joins a set. "The twelve
  studies that look like this one" is one relation over thirteen nodes, not 66
  pairwise edges that lose the fact they belong together.
- *Isn't transductive learning cheating?* It uses validation and test node
  **features**, never their labels — the standard HGNN setting. `transductive:
  false` gives the strictly inductive comparison, and both are reported.
- *How does the deployed model do hypergraph inference on one patient?* The
  frozen-bank extension, and here is the measured gap against the transductive
  computation.
- *Did the diffusion model help?* Point at the ablation row, whichever way it went.

---

## 7. Anti-goals

Things this project deliberately does **not** do, so you are not asked why they
are missing:

- No clinical claim of any kind. It is a research prototype, labelled as such on
  every surface.
- No attempt to beat a published MIMIC-CXR leaderboard number — those use the
  full 377k-image release and far more compute.
- No horizontal flip augmentation. Chest anatomy is not left–right symmetric;
  flipping invents dextrocardia.
- No patient data in the repository, the bundle, or the demo. The exported
  neighbour bank holds feature vectors and labels — never images or report text.
