# Runbook

Operational reference: commands, expected timings, and what to do when a stage
misbehaves. For *why* the project is built this way, read [`../PLAN.md`](../PLAN.md).

---

## Commands

```bash
export PYTHONPATH=src              # Windows cmd: set PYTHONPATH=src
                                   # Windows PowerShell: $env:PYTHONPATH = "src"

python -m dvlhg.cli info           # resolved config + which paths exist
python -m dvlhg.cli smoke          # whole pipeline on synthetic data (~3 min, CPU)

python -m dvlhg.cli prep           # manifest + image cache
python -m dvlhg.cli check-vlm      # zero-shot probe of the backbone
python -m dvlhg.cli diffusion      # stage 1: train the DDPM
python -m dvlhg.cli synth          # stage 1: generate synthetic rows
python -m dvlhg.cli train          # stage 2: vision-language backbone
python -m dvlhg.cli hypergraph     # stage 3 alone (eval does this too)
python -m dvlhg.cli eval           # stage 3+4: hypergraph, ablation, report, export
python -m dvlhg.cli verify-serving # does the API reproduce the report?
python -m dvlhg.cli serve          # stage 5: API + web app
```

All of them accept `--config path.yaml` and repeatable `--set key.sub=value`.

---

## Expected timings

Free Colab T4, `subset_size: 12000`, defaults everywhere else.

| stage | time | notes |
|---|---|---|
| `prep` (tables) | 2 min | ~40 MB |
| `prep` (12k images) | 20–40 min | resumable; the long pole |
| `check-vlm` | 2 min | first run also downloads ~800 MB of BiomedCLIP weights |
| `diffusion` | 35–50 min | 30 epochs at 128 px |
| `synth` | 8–12 min | 2000 samples at 50 DDIM steps |
| `train` | 8–15 min/epoch | 12 epochs ≈ 2–3 h |
| `eval` | 10–20 min | feature extraction dominates; the HGNN itself is seconds |
| `serve` | instant | model loads on first request, then warms up |

Single-study inference after warm-up: **~95 ms** on CPU, of which ~40 ms is the
backbone, ~4 ms the hypergraph query, ~45 ms Grad-CAM (skipped when
`explain=false`).

---

## Colab survival

1. **Put `paths.root` on Drive.** Everything else follows from this.
   ```python
   --set paths.root=/content/drive/MyDrive/dvlhg
   ```
2. **Re-run the same cell after a disconnect.** `diffusion` and `train` resume
   from the last completed epoch; `prep` skips files already downloaded.
3. **Do not reinstall torch.** Colab's build matches its CUDA driver; replacing
   it usually breaks the GPU for the session.
4. **One big file beats many small ones.** The image cache is a single `.npy`
   memmap for exactly this reason — 12k separate JPEGs take ~an hour to sync to
   Drive, one 800 MB file takes a couple of minutes.

---

## Troubleshooting

### `check-vlm` reports ~0.50 macro AUROC

The BiomedCLIP weights did not load; you are about to fine-tune random
initialisation. Re-run the install cell, confirm the machine can reach
huggingface.co, and check for a `create_model` error in the log. Fallback:

```bash
--set vlm.backbone=hf     # timm ViT-B/16 + Bio_ClinicalBERT
```

### CUDA out of memory during `train`

In order of preference:

```bash
--set train.batch_size=16 --set train.accum_steps=2   # same effective batch
--set vlm.unfreeze_vision_blocks=2
--set fusion.max_image_tokens=101                     # fewer patch tokens
--set fusion.token_level=false                        # pooled fusion only
```

### `prep` raises "manifest failed sanity checks"

It is refusing to hand you a leaky split. Read the listed reasons:

- *patients appear in two splits* — a custom split crossed a patient boundary;
  always split by `subject_id`
- *split X is empty* — `subset_size` is too small, or the filters removed everything
- *no positive case in split X* — raise `subset_size`, or keep `balance_subset: true`

### Diffusion samples look like static

The loss should fall steadily and end well under 0.1. If it does not:

```bash
--set diffusion.train.epochs=60
--set diffusion.image_size=64          # much easier target
--set diffusion.train.lr=0.0001        # if the loss is spiking
```

### Every diffusion sample in a row looks identical

Guidance has collapsed the model onto one mode:

```bash
--set diffusion.guidance_scale=1.5
```

### Macro AUROC above 0.95

Almost certainly text leakage. Check:

```bash
python -m dvlhg.cli info | grep text.mode      # should be 'indication'
```

`reports/report.md` also prints a banner on any leaky run. See
[`LEAKAGE.md`](LEAKAGE.md).

### `fusion_hypergraph` scores below `fusion`

The hypergraph is a residual on the backbone's logits, so this should be rare.
If it happens:

```bash
--set hypergraph.dropout=0.5
--set hgnn_train.lr=0.0003
--set hypergraph.edge_dropout=0.2
--set hypergraph.layers=1
```

If it still holds after three seeds, that is a result. Report it.

### The diffusion panel says "not in this bundle"

`export/diffusion.pt` is missing. Train one and re-export:

```bash
python -m dvlhg.cli diffusion
python -m dvlhg.cli eval --set serve.include_diffusion=true
```

The exported checkpoint is EMA weights only (no optimiser state), so it is about
a quarter the size of the training checkpoint.

### Every generated film looks the same whatever labels I tick

The diffusion model is undertrained, and the panel says so. A freshly
initialised UNet emits **exactly zero** — every ResBlock's second convolution and
the output convolution are zero-init by design — so the DDIM trajectory depends
only on the starting noise and the label vector has no effect at all.

`GET /api/capabilities` reports the measured `conditioning_strength`; below
`1e-4` the panel shows the warning. Train longer (notebook 02); the numbers move
once the residual branches leave zero.

### Grad-CAM is missing from the response

`grad_cam_multi` returns nothing when the backbone exposes no patch tokens.
Check that `fusion.token_level` is `true` — the pooled-only fusion path has no
spatial tokens to attribute to.

### The vision-language panel's alignment numbers look small or negative

Expected after fine-tuning. These are cosine similarities in the **fine-tuned**
backbone's space, not stock BiomedCLIP's: once the towers are trained for
classification they drift apart, and the prompts stop behaving like a zero-shot
classifier. For the pretrained model's zero-shot ability, run
`dvlhg check-vlm` before training instead.

### `verify-serving` fails

The API is not reproducing the evaluated model. In likelihood order:

1. **Preprocessing drift** — serving must go through `preprocess_image_pil` at
   `cache_size`, then resize to `image_size`. Both paths share that function; if
   you edited one, check the other.
2. **Stale bundle** — `export/` predates the last `eval`. Re-run `eval`.
3. **Text mode mismatch** — `bundle.json`'s `text_mode` differs from the config
   used for evaluation.
4. **Thresholds/temperatures** — confirm they are in `bundle.json` and non-trivial.

### The demo returns near-identical probabilities for every image

Same cause as above: the model is being fed off-distribution inputs. Run
`verify-serving` — that is exactly what it is for.

### Report tokens show numbers instead of words

Expected with `vlm.backbone=dummy` (the hashing tokenizer is not invertible, so
the raw words are shown instead). With BiomedCLIP you get real word-pieces.

---

## Reproducing a run

Every stage writes the exact config it used next to its checkpoint:

```
checkpoints/diffusion_config.yaml
checkpoints/backbone_config.yaml
checkpoints/hypergraph_config.yaml
export/config.yaml
```

To reproduce:

```bash
python -m dvlhg.cli eval --config runs/<run>/export/config.yaml
```

`reports/metrics.json` additionally records the data source, split sizes, text
mode and leakage flag, backbone, hypergraph summary, learned edge weights and
the served-vs-evaluated check.

---

## Safety checklist before showing anyone the demo

- [ ] The disclaimer is visible on the page (it is, by default — do not remove it)
- [ ] No identifiable patient data has been uploaded
- [ ] The Cloudflare tunnel is stopped when you are finished — it is public and
      unauthenticated for as long as it runs
- [ ] `verify-serving` passes, so the demo and the report agree
- [ ] The model card loads, showing the test metrics and the known limitations
