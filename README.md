# DVL-HGN — Diffusion-Augmented Vision-Language Hypergraph Network

Chest radiograph + radiology report → four findings:
**Atelectasis · Cardiomegaly · Edema · Pleural Effusion**

> **Research prototype. Not a medical device.** Not validated for clinical use.
> Nothing here may be used to diagnose, treat, or make decisions about a patient.

```
MIMIC-CXR ─► preprocessing ─► diffusion augmentation ─► BiomedCLIP (image + text)
          ─► cross-attention fusion ─► hypergraph neural network ─► 4 predictions
          ─► evaluation (AUROC / accuracy / precision / recall / F1) ─► web app
```

**[`PLAN.md`](PLAN.md) is the place to start** — what to build, in what order,
what will go wrong, and how to tell whether it worked.

---

## Quickstart — 5 minutes, no data, no GPU, no accounts

```bash
git clone <this repo> && cd dvl-hypergraph
pip install -r requirements.txt
export PYTHONPATH=src            # Windows: set PYTHONPATH=src

python -m dvlhg.cli smoke --n 900 --epochs 8
```

This runs **every stage** — preprocessing, diffusion, synthesis, the
vision-language backbone, the hypergraph, evaluation and export — on
procedurally generated chest films that carry real, label-correlated structure.
It finishes on a laptop CPU in a few minutes and prints:

```
================ SMOKE TEST RESULT ================
  image_only           test macro AUROC 0.6299
  text_only            test macro AUROC 0.6586
  fusion               test macro AUROC 0.7342
  fusion_hypergraph    test macro AUROC 0.7419
  inductive vs transductive mean |dp| = 0.0000
===================================================
```

Then serve it:

```bash
python -m dvlhg.cli serve --bundle runs/smoke/export   # http://127.0.0.1:8000
```

## On Colab

Six notebooks, numbered, each runnable top to bottom:

| notebook | does | time |
|---|---|---|
| `00_quickstart_smoke` | the whole pipeline on synthetic data | 5 min |
| `01_data_and_manifest` | manifest + image cache + backbone sanity check | 30–60 min |
| `02_diffusion` | train the diffusion model, generate synthetic rows | 45–70 min |
| `03_backbone` | train the vision-language backbone | 2–3 h |
| `04_hypergraph_eval_export` | hypergraph, ablation, report, bundle | 20 min |
| `05_demo_serve` | public demo through a Cloudflare tunnel | 5 min |

Put `paths.root` on Google Drive and every stage survives a disconnect — each
epoch writes a full resumable checkpoint.

Regenerate the notebooks after editing `scripts/build_notebooks.py`:

```bash
python scripts/build_notebooks.py
python scripts/make_colab_zip.py     # if you upload rather than git clone
```

## Data

| source | access | size | reports | labels |
|---|---|---|---|---|
| `mimic` | PhysioNet **credentialed** | 377k studies | yes | CheXpert labeler |
| `openi` | public, no account | ~7.4k | yes | MeSH terms |
| `synthetic` | none | generated | templated | exact |

MIMIC-CXR-JPG access takes days to approve — apply first, and develop against
`openi` meanwhile. Everything downstream is identical; only the numbers change.

```bash
export PHYSIONET_USER=... PHYSIONET_PASS=...
python -m dvlhg.cli prep --set data.source=mimic --set data.subset_size=12000
```

Only the studies in the manifest are downloaded — never the full 570 GB release.

## The pipeline

```bash
python -m dvlhg.cli prep             # manifest + image cache
python -m dvlhg.cli check-vlm        # did BiomedCLIP really load? (zero-shot probe)
python -m dvlhg.cli diffusion        # stage 1: train the DDPM
python -m dvlhg.cli synth            # stage 1: generate synthetic training rows
python -m dvlhg.cli train            # stage 2: vision-language backbone
python -m dvlhg.cli eval             # stage 3+4: hypergraph, ablation, report, export
python -m dvlhg.cli verify-serving   # does the API reproduce the report?
python -m dvlhg.cli serve            # stage 5: API + web app
```

Every command takes `--config` and repeatable `--set key.sub=value`:

```bash
python -m dvlhg.cli train --set train.batch_size=16 --set train.accum_steps=2
```

## Three things this project takes seriously

**Label provenance.** MIMIC's labels were produced by an NLP tool reading
FINDINGS and IMPRESSION. Feeding those sections back in scores ~0.98 AUROC and
means nothing. The default `text.mode` is `indication` — pre-read clinical
context only — and the leaky modes exist, clearly named, so the gap can be
measured. → [`docs/LEAKAGE.md`](docs/LEAKAGE.md)

**Hypergraph inference on one patient.** An HGNN is transductive; a deployed
model gets one study at a time. The frozen-bank protocol makes single-study
inference exact, ~4 ms, and stable — one patient's prediction can never shift
another's — and `dvlhg eval` measures the gap against the transductive
computation instead of assuming it away. → [`docs/HYPERGRAPH.md`](docs/HYPERGRAPH.md)

**Train/serve agreement.** `dvlhg verify-serving` pushes real test studies through
the live API and diffs the result against the exact predictions the report quotes:

```
  mean |dp|               0.000081
  macro AUROC  served     0.7052
  macro AUROC  reported   0.7070
  PASS - the API reproduces the evaluated model (tolerance 0.01)
```

## Layout

```
configs/default.yaml          every knob, commented
src/dvlhg/
  data/       manifest builders (mimic · openi · synthetic), text sections, dataset
  diffusion/  conditional UNet, DDPM/DDIM, SDEdit refinement, synthesis
  vlm/        BiomedCLIP / timm+HF / offline-dummy backbones behind one interface
  fusion/     bidirectional cross-attention + gated pooling
  hypergraph/ hyperedge construction, HGNN operator, inductive query path
  models/     Backbone (stage 2) and HypergraphClassifier (stage 3)
  train/      losses (masked BCE, asymmetric), backbone loop, HGNN trainer
  eval/       metrics, report, figures, explanations
  serve/      FastAPI + single-study inference
frontend/index.html           the web app (served at /)
notebooks/                    six Colab notebooks
tests/                        pytest suite
docs/                         LEAKAGE.md · HYPERGRAPH.md · RUNBOOK.md
```

## Tests

```bash
pip install pytest && PYTHONPATH=src python -m pytest tests -q
```

The metric implementations are checked against scikit-learn to 1e-9 (including
tie handling and masked cells), the HGNN operator against a dense reference, and
the inductive query path against the transductive forward.

## Citing the pieces

- Johnson et al., *MIMIC-CXR-JPG* (2019) — PhysioNet
- Irvin et al., *CheXpert* (2019) — the labeler behind the labels
- Zhang et al., *BiomedCLIP* (2023) — the vision-language backbone
- Feng et al., *Hypergraph Neural Networks*, AAAI (2019) — the HGNN operator
- Ho et al., *DDPM* (2020); Song et al., *DDIM* (2021); Meng et al., *SDEdit* (2022)
- Ridnik et al., *Asymmetric Loss for Multi-Label Classification* (2021)
- Demner-Fushman et al., *Open-i / Indiana University CXR* (2016)

## License and use

Research and educational use. MIMIC-CXR carries its own data use agreement —
read it, and do not redistribute images, reports, or anything derived from them
that could identify a patient. The exported bundle deliberately contains feature
vectors and labels only.
