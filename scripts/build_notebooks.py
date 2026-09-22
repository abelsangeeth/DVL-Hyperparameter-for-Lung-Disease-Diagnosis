"""Generate the Colab notebooks in notebooks/.

Keeping the notebooks generated rather than hand-edited means the setup cell,
the Drive paths and the CLI invocations stay identical across all five of them.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

REPO_URL = "https://github.com/abelsangeeth/DVL-Hyperparameter-for-Lung-Disease-Diagnosis.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> List[str]:
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def notebook(cells: List[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# --------------------------------------------------------------------------- #
# shared cells
# --------------------------------------------------------------------------- #
BANNER = """
> **Research prototype — not a medical device.** Nothing produced by these notebooks may be
> used to diagnose, treat, or make any decision about a patient.
"""

GPU_CELL = """
import subprocess, sys
print(sys.version)
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "no GPU reported")
except FileNotFoundError:
    print("nvidia-smi not found - you are on CPU. Runtime > Change runtime type > T4 GPU.")
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
"""

SETUP_CELL = f"""
# --- 1. where results live -------------------------------------------------
# Mounting Drive is strongly recommended: Colab disconnects, and every stage
# here writes a resumable checkpoint. Without Drive you start over.
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_ROOT = '/content/drive/MyDrive/dvlhg'
else:
    RUN_ROOT = '/content/dvlhg'

# --- 2. get the code -------------------------------------------------------
# Pick ONE. 'clone' is easiest once you have pushed this repo to GitHub.
SOURCE = 'clone'        # 'clone' | 'zip' | 'drive'
REPO_URL = '{REPO_URL}'
ZIP_PATH = '/content/dvl-hypergraph.zip'          # if SOURCE == 'zip'
DRIVE_CODE = '/content/drive/MyDrive/dvl-hypergraph'  # if SOURCE == 'drive'

import os, shutil, subprocess, sys
CODE = '/content/dvl-hypergraph'
if not os.path.exists(CODE):
    if SOURCE == 'clone':
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, CODE], check=True)
    elif SOURCE == 'zip':
        if not os.path.exists(ZIP_PATH):
            from google.colab import files
            up = files.upload()            # choose the zip from scripts/make_colab_zip.py
            ZIP_PATH = '/content/' + next(iter(up))
        shutil.unpack_archive(ZIP_PATH, '/content/')
    elif SOURCE == 'drive':
        shutil.copytree(DRIVE_CODE, CODE)
print('code at', CODE, '| contents:', sorted(os.listdir(CODE))[:8])

# --- 3. dependencies -------------------------------------------------------
# Colab already ships torch/torchvision built for its CUDA - never reinstall them.
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                'open_clip_torch>=2.24', 'timm>=0.9.12', 'transformers>=4.35',
                'fastapi', 'uvicorn', 'python-multipart'], check=True)

sys.path.insert(0, os.path.join(CODE, 'src'))
os.chdir(CODE)
os.environ['PYTHONPATH'] = os.path.join(CODE, 'src')
os.environ['RUN_ROOT'] = RUN_ROOT
print('run root ->', RUN_ROOT)
"""

CONFIG_CELL = """
# Everything below is overridden on the command line, so this cell is the only
# place you need to edit. Values here are tuned for a Colab T4.
CFG = dict(
    subset_size = 12000,     # frontal studies pulled from MIMIC-CXR-JPG
    text_mode   = 'indication',   # see docs/LEAKAGE.md before changing this
    batch_size  = 24,
    epochs      = 12,
    diffusion_epochs = 30,
    synth_per_class  = 500,
)

def dvlhg(command, **overrides):
    \"\"\"Run a dvlhg subcommand with RUN_ROOT and any overrides applied.\"\"\"
    import os, shlex, subprocess, sys
    args = [sys.executable, '-m', 'dvlhg.cli', *shlex.split(command)]
    args += ['--set', f"paths.root={os.environ['RUN_ROOT']}"]
    for key, value in overrides.items():
        args += ['--set', f'{key}={value}']
    print('$', ' '.join(args[2:]))
    return subprocess.run(args, check=True)
"""


# --------------------------------------------------------------------------- #
# notebooks
# --------------------------------------------------------------------------- #
def nb_quickstart() -> dict:
    return notebook([
        md(f"""
# 00 · Quickstart — prove the pipeline works (3 minutes, no data needed)

{BANNER}

This notebook runs **every stage** — preprocessing, diffusion, synthesis, the
vision-language backbone, the hypergraph, evaluation and export — on
procedurally generated films. No PhysioNet account, no downloads, no GPU.

Run it first. If it finishes, your environment is correct and any later problem
is about the data, not the code.
"""),
        code(GPU_CELL),
        code(SETUP_CELL),
        md("""
### The smoke run

`dvlhg smoke` builds ~900 synthetic chest films with real, label-correlated
structure (an enlarged cardiac silhouette for Cardiomegaly, a blunted
costophrenic angle for Pleural Effusion, a basal band for Atelectasis,
perihilar haze for Edema), then runs the whole pipeline on them.

The four ablation numbers it prints at the end should be **above chance**. They
are not meaningful clinical numbers — the point is that every component
connects and learns.
"""),
        code("""
import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'dvlhg.cli', 'smoke',
                '--n', '900', '--epochs', '8',
                '--root', os.path.join(os.environ['RUN_ROOT'], 'smoke')], check=True)
"""),
        md("### What the synthetic films look like"),
        code("""
import numpy as np, matplotlib.pyplot as plt
from dvlhg.data.synthetic import render_cxr
from dvlhg.constants import LABELS

rng = np.random.default_rng(0)
cases = [np.zeros(4, np.float32)] + [np.eye(4, dtype=np.float32)[i] for i in range(4)]
names = ['no finding'] + LABELS
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for ax, vector, name in zip(axes, cases, names):
    ax.imshow(render_cxr(vector, 256, rng), cmap='gray', vmin=0, vmax=1)
    ax.set_title(name, fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
"""),
        md("""
### Next

- **01** prepares the real dataset (MIMIC-CXR-JPG, or Open-i if you have no PhysioNet access)
- **02** trains the diffusion model
- **03** trains the vision-language backbone
- **04** builds the hypergraph, evaluates, and exports the serving bundle
- **05** runs the web demo
"""),
    ])


def nb_data() -> dict:
    return notebook([
        md(f"""
# 01 · Data — manifest, image cache, and a backbone sanity check

{BANNER}

## Which dataset?

| source | access | images | reports | labels |
|---|---|---|---|---|
| `mimic` | PhysioNet **credentialed** (CITI course + signed DUA) | 377k | yes | CheXpert labeler |
| `openi` | public, no account | ~7.4k | yes | MeSH terms |
| `synthetic` | none | generated | templated | exact |

MIMIC-CXR-JPG is the right dataset for this project. Getting access takes a few
days: complete the CITI "Data or Specimens Only Research" course, upload the
certificate to PhysioNet, then sign the MIMIC-CXR-JPG data use agreement.

**While you wait, use `openi`** — it is a real chest X-ray dataset with real
reports, it needs no credentials, and every later notebook works unchanged.

## What this notebook does not do

It does **not** download the full 570 GB release. It fetches three small
metadata tables, decides which `subset_size` studies to use, and then downloads
only those images.
"""),
        code(GPU_CELL),
        code(SETUP_CELL),
        code(CONFIG_CELL),
        md("""
### Credentials

Typed into a password box, not stored in the notebook. Skip this cell entirely
if you are using `openi`.
"""),
        code("""
import getpass, os
SOURCE_DATASET = 'mimic'     # 'mimic' | 'openi' | 'synthetic'

if SOURCE_DATASET == 'mimic':
    os.environ['PHYSIONET_USER'] = input('PhysioNet username: ')
    os.environ['PHYSIONET_PASS'] = getpass.getpass('PhysioNet password: ')
    print('credentials set for this session only')
"""),
        md("""
### Build the manifest and the image cache

This is the long step — downloading ~12k JPEGs takes roughly 20–40 minutes on
Colab. It is resumable: files already present are skipped, so a disconnect
costs nothing.

The image cache is written as **one** `.npy` memmap rather than 12k small
files, because copying 12k files to Drive takes the better part of an hour and
copying one 800 MB file takes two minutes.
"""),
        code("""
dvlhg('prep',
      **{'data.source': SOURCE_DATASET,
         'data.subset_size': CFG['subset_size'],
         'text.mode': CFG['text_mode']})
"""),
        md("### What did we get?"),
        code("""
import json, os, pandas as pd
from dvlhg.config import load_config
from dvlhg.data.prepare import load_manifest

cfg = load_config('configs/default.yaml',
                  [f"paths.root={os.environ['RUN_ROOT']}",
                   f"data.source={SOURCE_DATASET}", f"text.mode={CFG['text_mode']}"])
manifest = load_manifest(cfg)
stats = json.load(open(os.path.join(cfg.paths.manifest, 'stats.json')))

print('rows:', len(manifest), '| splits:', stats['splits'])
print('text mode:', stats['text_mode'], '-', stats['text_mode_description'])
print('leaky:', stats['text_is_leaky'])
print()
print(pd.DataFrame(stats['label_stats_all']).T.round(3))
print()
print('example text the model will see:')
print(' ', manifest['text'].iloc[0][:300])
"""),
        md("""
### Class balance and co-occurrence

The co-occurrence matrix is worth a look — it is the structure the hypergraph's
prototype hyperedges are meant to exploit.
"""),
        code("""
import numpy as np, matplotlib.pyplot as plt
from dvlhg.constants import LABELS
from dvlhg.data.labels import targets_from_manifest

y, mask = targets_from_manifest(manifest)
counts = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        both = (mask[:, i] > 0) & (mask[:, j] > 0)
        counts[i, j] = ((y[both, i] > .5) & (y[both, j] > .5)).sum()

fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(counts, cmap='Blues')
ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=40, ha='right', fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(LABELS, fontsize=8)
for i in range(4):
    for j in range(4):
        ax.text(j, i, int(counts[i, j]), ha='center', va='center', fontsize=8,
                color='white' if counts[i, j] > counts.max()/2 else '#0b0b0b')
ax.set_title('co-occurrence of positive labels', fontsize=10, loc='left')
plt.tight_layout(); plt.show()
"""),
        md("""
### Sanity check the backbone **before** spending GPU hours

`check-vlm` runs BiomedCLIP zero-shot: it scores each film against
"chest x-ray showing {finding}" versus "chest x-ray with no {finding}" and
reports AUROC. No training involved.

A correctly loaded BiomedCLIP lands somewhere around 0.6–0.8 here. If you see
~0.50, the weights did not download and you would be fine-tuning random
initialisation for an hour without noticing.
"""),
        code("""
dvlhg('check-vlm --limit 600',
      **{'data.source': SOURCE_DATASET, 'text.mode': CFG['text_mode']})
"""),
        md("""
### Next

**02 · Diffusion.** If you want to skip the diffusion stage entirely, go
straight to **03** and pass `--set train.use_synthetic=false`.
"""),
    ])


def nb_diffusion() -> dict:
    return notebook([
        md(f"""
# 02 · Diffusion — learned augmentation

{BANNER}

## What this stage is for

Two jobs, both configured under `diffusion.use`:

**`synthesis`** — sample new films conditioned on a label vector and add them to
the training split. Label vectors are resampled from the *training* label
distribution conditioned on each finding in turn, so co-occurrence stays
realistic while every finding gets an equal budget. This is aimed squarely at
the rare combinations a 12k subset barely covers.

**`refine`** — SDEdit. Noise a real film to ~25% of the schedule and denoise it
back. Anatomy survives; texture and acquisition characteristics change. It is a
learned augmentation that respects the image prior instead of jittering pixels.

## Honest scoping

This is a **128 px, ~22M parameter** UNet trained for ~30 epochs on ~10k films.
It is not RoentGen and it will not produce diagnostic-quality radiographs. It is
sized to finish on a free T4 in under an hour.

The samples are therefore used **as extra training rows with a reduced loss
weight** (`train.synthetic_weight: 0.5`), never as a replacement for real data.
Whether they help is an empirical question that notebook 04 answers with an
ablation — do not assume they do.

**The generator sees the training split only.** A model that had seen validation
or test films would launder them back into training and every later number would
be meaningless.
"""),
        code(GPU_CELL),
        code(SETUP_CELL),
        code(CONFIG_CELL),
        code("""
SOURCE_DATASET = 'mimic'   # keep this the same across all notebooks
"""),
        md("""
### Train

~35–50 minutes on a T4 at the defaults. Resumable: re-run the cell after a
disconnect and it continues from the last completed epoch.
"""),
        code("""
dvlhg('diffusion',
      **{'data.source': SOURCE_DATASET,
         'text.mode': CFG['text_mode'],
         'diffusion.train.epochs': CFG['diffusion_epochs']})
"""),
        md("### Look at the samples before trusting them"),
        code("""
import os
from IPython.display import Image as ShowImage, display
path = os.path.join(os.environ['RUN_ROOT'], 'reports', 'figures', 'diffusion_samples.png')
display(ShowImage(filename=path))
"""),
        md("""
Judge these on three things, in order:

1. **Anatomy** — two lung fields, a mediastinum, a diaphragm, ribs. If the
   samples are noise, train longer or lower `diffusion.image_size`.
2. **Conditioning** — does the Cardiomegaly row have a visibly wider cardiac
   silhouette than the "no finding" row? If not, raise
   `diffusion.guidance_scale` (try 3.0–4.0).
3. **Diversity** — if every sample in a row looks identical, guidance is too
   high and the model has collapsed onto one mode.

Only the downstream ablation in notebook 04 tells you whether they *help*.
"""),
        md("### Generate the synthetic training rows"),
        code("""
dvlhg('synth',
      **{'data.source': SOURCE_DATASET,
         'text.mode': CFG['text_mode'],
         'diffusion.synth_per_class': CFG['synth_per_class']})
"""),
        code("""
import json, os
stats = json.load(open(os.path.join(os.environ['RUN_ROOT'], 'synth', 'synth_stats.json')))
print(json.dumps(stats, indent=2))
"""),
    ])


def nb_backbone() -> dict:
    return notebook([
        md(f"""
# 03 · Vision-language backbone

{BANNER}

## Architecture

```
film  ──► BiomedCLIP ViT-B/16 ──► 197 patch tokens ─┐
                                                    ├─► bidirectional cross-attention
report ─► PubMedBERT ──────────► 256 text tokens ───┘   (2 blocks) ─► gated pool
                                                                        │
                                                                   node feature (384-d)
                                                                        │
                                                            ┌───────────┴───────────┐
                                                        4 logits            (stage 3: hypergraph)
```

BiomedCLIP is pretrained on 15M biomedical image–text pairs, so the two towers
already share a space — the fusion trunk does not have to learn alignment from
12k studies.

## What is trained

Only the **last 4 ViT blocks** and the **last 2 BERT blocks**, plus all norms,
the projections, the fusion trunk and the heads. Full fine-tuning of both towers
overfits this subset and will not fit a T4 at a useful batch size.

## Two details that matter

**No horizontal flip.** Chest anatomy is not left–right symmetric; flipping a
film invents dextrocardia and destroys the cue the Cardiomegaly head needs.

**Modality dropout (15%).** The report is blanked on 15% of training steps, so
the model stays usable when the demo gets an image with no report. Without it,
an image-only request falls off a cliff.
"""),
        code(GPU_CELL),
        code(SETUP_CELL),
        code(CONFIG_CELL),
        code("""
SOURCE_DATASET = 'mimic'
"""),
        md("""
### Train

~8–15 minutes per epoch on a T4 at batch 24. **Every epoch writes a full
checkpoint** (weights, optimiser, scheduler, scaler, history) — if Colab drops,
re-run this cell and it resumes from the last completed epoch.

If you hit CUDA OOM: lower `train.batch_size` to 16 and raise
`train.accum_steps` to 2 (same effective batch), or set
`vlm.unfreeze_vision_blocks=2`.
"""),
        code("""
dvlhg('train',
      **{'data.source': SOURCE_DATASET,
         'text.mode': CFG['text_mode'],
         'train.epochs': CFG['epochs'],
         'train.batch_size': CFG['batch_size']})
"""),
        md("### Training curves"),
        code("""
import json, os, matplotlib.pyplot as plt
history = json.load(open(os.path.join(os.environ['RUN_ROOT'], 'logs', 'backbone_history.json')))
epochs = [h['epoch'] for h in history]

fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
axes[0].plot(epochs, [h['train_loss'] for h in history], color='#2a78d6', lw=2)
axes[0].set_ylabel('train loss'); axes[0].grid(color='#e6e5e1')
axes[1].plot(epochs, [h['val_auroc_macro'] for h in history], color='#2a78d6', lw=2,
             marker='o', ms=4, label='val macro AUROC')
axes[1].plot(epochs, [h['val_f1_macro'] for h in history], color='#eb6834', lw=2,
             marker='s', ms=4, ls='--', label='val macro F1')
axes[1].set_xlabel('epoch'); axes[1].set_ylabel('score'); axes[1].grid(color='#e6e5e1')
axes[1].legend(frameon=False)
for ax in axes:
    for side in ('top', 'right'): ax.spines[side].set_visible(False)
plt.tight_layout(); plt.show()

best = max(history, key=lambda h: h['val_auroc_macro'])
print(f"best epoch {best['epoch']}: val macro AUROC {best['val_auroc_macro']:.4f}")
"""),
        md("""
**Reading the curves.** Validation AUROC plateauing while train loss keeps
falling is normal and the early-stopping rule handles it. Validation AUROC
*falling* for several epochs means the encoder learning rate is too high —
lower `train.encoder_lr_scale` from 0.1 to 0.05.
"""),
    ])


def nb_hypergraph() -> dict:
    return notebook([
        md(f"""
# 04 · Hypergraph, evaluation, export

{BANNER}

## Why a hypergraph

A graph edge joins two nodes. A **hyperedge joins a set**. "The twelve studies
that look like this one" is naturally one relation over thirteen nodes, not
sixty-six pairwise edges that lose the fact they belong together.

Four families of hyperedge are built over the frozen node features:

| group | what it connects | built from |
|---|---|---|
| `knn_fused` | `{{i}} ∪ kNN(i)` in fused space | features |
| `knn_image` | kNN in image space only | features |
| `knn_text` | kNN in report space only | features |
| `meta_view` | studies sharing PA/AP/... | metadata |
| `meta_demo` | studies sharing (sex, age decade) | metadata, when available |
| `proto_label` | k-means prototypes of **training** positives | train labels only |

The layer is the standard HGNN operator

$$X' = \\sigma\\left(D_v^{{-1/2}} H W D_e^{{-1}} H^\\top D_v^{{-1/2}} X \\Theta\\right)$$

computed as two sparse mat-muls, never as an N×N matrix.

## Leakage rules, enforced in code

- Labels are read **only** for rows whose split is `train`.
- Validation and test nodes join `proto_label` edges by *feature distance*, never
  by their own label.
- Validation and test nodes do take part in message passing. That is the standard
  transductive HGNN setting and it uses their **features**, not their labels. Set
  `hypergraph.transductive=false` for a strictly inductive comparison.

## Why the backbone is frozen here

The hypergraph is transductive: it needs every node's feature at once, which
cannot be held on a T4 while gradients flow through two transformers. Trained on
frozen features, an epoch is milliseconds — which is what makes it possible to
actually ablate the design instead of asserting it helps.
"""),
        code(GPU_CELL),
        code(SETUP_CELL),
        code(CONFIG_CELL),
        code("""
SOURCE_DATASET = 'mimic'
"""),
        md("""
### Run evaluation

This extracts node features, builds the hypergraph, trains the HGNN, computes
the ablation table with patient-level bootstrap CIs, checks the served path
against the evaluated one, writes the report and figures, and exports the
serving bundle.
"""),
        code("""
dvlhg('eval',
      **{'data.source': SOURCE_DATASET, 'text.mode': CFG['text_mode']})
"""),
        md("### The ablation table"),
        code("""
import json, os, pandas as pd
metrics = json.load(open(os.path.join(os.environ['RUN_ROOT'], 'reports', 'metrics.json')))

rows = []
for name, entry in metrics['results'].items():
    test = entry['test']
    ci = entry.get('ci', {}).get('auroc_macro')
    rows.append({
        'variant': name,
        'AUROC': round(test['auroc_macro'], 4),
        '95% CI': f"{ci['lo']:.3f}-{ci['hi']:.3f}" if ci else '-',
        'AUPRC': round(test['auprc_macro'], 4),
        'F1': round(test['f1_macro'], 4),
        'accuracy': round(test['accuracy_macro'], 4),
        'exact match': round(test['exact_match'], 4),
    })
display(pd.DataFrame(rows).set_index('variant'))

check = metrics.get('inductive_check')
if check:
    print(f"\\nserved (frozen-bank) vs evaluated (transductive) on {check['n']} studies:"
          f" mean |dp| = {check['mean_abs_prob_delta']:.4f},"
          f" AUROC {check['auroc_inductive']:.3f} vs {check['auroc_transductive']:.3f}")
"""),
        md("""
**How to read this.** `fusion_hypergraph` beating `fusion` by less than the
width of its confidence interval is *not* evidence the hypergraph helps. Say so
if that is what you see — a well-reported null result is worth more than an
overstated gain.

The `image_only` vs `text_only` gap is the other thing to look at. If
`text_only` is close to `fusion`, the model is leaning on the report; check
`text.mode` is `indication` and not one of the leaky modes.
"""),
        md("### Per-finding results"),
        code("""
best = 'fusion_hypergraph' if 'fusion_hypergraph' in metrics['results'] else 'fusion'
per_class = metrics['results'][best]['test']['per_class']
frame = pd.DataFrame(per_class).T[
    ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'specificity', 'f1',
     'threshold', 'n', 'n_positive', 'prevalence']].round(3)
display(frame)

for name in per_class:
    ci = metrics['results'][best].get('ci', {}).get(f'auroc::{name}')
    if ci:
        print(f"{name:20s} AUROC {per_class[name]['auroc']:.3f}  (95% CI {ci['lo']:.3f}-{ci['hi']:.3f})")
"""),
        md("### Figures"),
        code("""
import os
from IPython.display import Image as ShowImage, display
figures = os.path.join(os.environ['RUN_ROOT'], 'reports', 'figures')
for name in ['ablation.png', 'roc_curves.png', 'pr_curves.png',
             'hyperedge_weights.png', 'training_history.png']:
    path = os.path.join(figures, name)
    if os.path.exists(path):
        print(name); display(ShowImage(filename=path))
"""),
        md("""
### The learned hyperedge weights are a result in themselves

Each group starts at weight 1.0. Where the model ends up tells you which
relations carried information — if `knn_text` ends far above `meta_demo`, the
report-space neighbourhood mattered and the demographic bucket did not. That is
a finding worth a sentence in the write-up.
"""),
        code("""
print(json.dumps(metrics.get('hyperedge_group_weights', {}), indent=2))
print()
print(json.dumps(metrics.get('hypergraph', {}), indent=2))
"""),
        md("### The written report"),
        code("""
import os
from IPython.display import Markdown, display
display(Markdown(open(os.path.join(os.environ['RUN_ROOT'], 'reports', 'report.md'),
                      encoding='utf-8').read()))
"""),
        md("""
### Download the serving bundle

Everything notebook 05 and the standalone API need: the backbone weights, the
HGNN weights, the neighbour bank, the thresholds and the model card.

The bank contains **training and validation** nodes only — no test film is ever
shipped, and a query can never retrieve itself.
"""),
        code("""
import os, shutil
export = os.path.join(os.environ['RUN_ROOT'], 'export')
print(sorted(os.listdir(export)))
archive = shutil.make_archive('/content/dvlhg_bundle', 'zip', export)
print('size: %.1f MB' % (os.path.getsize(archive) / 1e6))
from google.colab import files
files.download(archive)
"""),
    ])


def nb_serve() -> dict:
    return notebook([
        md(f"""
# 05 · Demo — the web app

{BANNER}

Runs the FastAPI service and the single-page frontend from inside Colab, exposed
through a Cloudflare quick tunnel.

The page has three panels beyond the predictions themselves:

- **Grad-CAM** — click any finding to see the heatmap for *that* finding. All four
  come back from one forward pass, so switching is instant.
- **Diffusion** — generate a film from any label combination with live guidance and
  DDIM-step controls, and SDEdit-refine your own upload. Needs `diffusion.pt` in
  the bundle (notebook 02, then re-run `dvlhg eval`).
- **Vision-language** — image-text alignment per finding, and the same film re-run
  with the report blanked so you can read off what the language channel adds.

**Anyone with that URL can reach your model.** It is a public tunnel with no
authentication. Do not upload identifiable patient data, and stop the tunnel
when you are finished.
"""),
        code(SETUP_CELL),
        md("### Point at the bundle"),
        code("""
import os
BUNDLE = os.path.join(os.environ['RUN_ROOT'], 'export')
print('bundle:', BUNDLE, '|', sorted(os.listdir(BUNDLE)))
"""),
        md("### Start the API"),
        code("""
import os, subprocess, sys, time

env = dict(os.environ,
           DVLHG_BUNDLE=BUNDLE, DVLHG_DEVICE='auto', DVLHG_NEIGHBOURS='6',
           PYTHONPATH=os.path.join(os.getcwd(), 'src'))
server = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'dvlhg.serve.api:app', '--host', '0.0.0.0', '--port', '8000'],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

import urllib.request
for attempt in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)
        print('server up'); break
    except Exception:
        time.sleep(1)
else:
    print(server.stdout.read()[-3000:])
"""),
        md("### Open a public tunnel"),
        code("""
import re, subprocess, time
subprocess.run('wget -q -O /usr/local/bin/cloudflared '
               'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'
               ' && chmod +x /usr/local/bin/cloudflared', shell=True, check=True)

tunnel = subprocess.Popen(['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000', '--no-autoupdate'],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
url = None
for line in tunnel.stdout:
    match = re.search(r'https://[-\\w]+\\.trycloudflare\\.com', line)
    if match:
        url = match.group(0); break
print('\\n  OPEN THIS:', url, '\\n')
print('  Remember: public URL, no authentication. Stop it when you are done.')
"""),
        md("""
### Try it without leaving the notebook

The call below posts a film straight to the local API — useful for checking the
response shape before opening the UI.
"""),
        code("""
import io, json, numpy as np, requests
from PIL import Image
from dvlhg.data.synthetic import render_cxr, make_report

rng = np.random.default_rng(1)
labels = np.array([0., 1., 0., 1.], np.float32)   # cardiomegaly + effusion
buffer = io.BytesIO()
Image.fromarray((render_cxr(labels, 256, rng) * 255).astype('uint8'), 'L').save(buffer, 'PNG')

response = requests.post('http://127.0.0.1:8000/api/predict',
                         files={'image': ('film.png', buffer.getvalue(), 'image/png')},
                         data={'report': make_report(labels, rng), 'explain': 'true'})
result = response.json()
for finding in result['findings']:
    flag = 'FLAG' if finding['flagged'] else '    '
    print(f"  {flag} {finding['label']:18s} {finding['probability']:.3f}"
          f"  (threshold {finding['threshold']:.2f}, without hypergraph {finding['without_hypergraph']:.3f})")
print('\\nlatency', result['latency_ms'], 'ms | modality gate', result['modality_gate'])
print('neighbours:', [(n['similarity'], n['findings']) for n in result.get('neighbours', [])[:3]])
"""),
        md("### Shut down"),
        code("""
tunnel.terminate(); server.terminate()
print('tunnel and server stopped')
"""),
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_quickstart_smoke.ipynb": nb_quickstart(),
        "01_data_and_manifest.ipynb": nb_data(),
        "02_diffusion.ipynb": nb_diffusion(),
        "03_backbone.ipynb": nb_backbone(),
        "04_hypergraph_eval_export.ipynb": nb_hypergraph(),
        "05_demo_serve.ipynb": nb_serve(),
    }
    for name, content in notebooks.items():
        path = OUT / name
        path.write_text(json.dumps(content, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {path} ({len(content['cells'])} cells)")


if __name__ == "__main__":
    main()
