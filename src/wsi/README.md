# `wsi` — your method goes here

This package is where the **new** methods are developed. Everything under `scripts/` is
the *evaluation harness* and *competitor baselines* (cellina, CPA); `src/wsi/` is yours.

## The idea

Bring **whole-slide-image (WSI)** information into the `cellina` spatial-perturbation
model. `cellina` disentangles a cell's intrinsic state `z` from its transcriptomic
spatial context `s`. We want to additionally use features derived from the H&E / WSI
image of the same tissue — e.g. tile embeddings from [LazySlide](https://lazyslide.readthedocs.io/)
— as a second view of the microenvironment.

## Suggested first steps

1. **Get the data lined up.** Work through `notebooks/01_spatialdata_load.ipynb` to load
   the CRC sample as a `SpatialData` object (transcriptomics table + registered image),
   and `notebooks/02_lazyslide.ipynb` to extract per-tile / per-cell image features.
2. **Establish the baseline.** Run `notebooks/loo_eval.ipynb` to reproduce the
   cellina-vs-CPA leave-one-cell-type-out comparison (held-out: Myeloid, sample
   `crc_232`). This is the number your method needs to beat.
3. **Implement `WSICellina`** in `model.py`. Prefer *wrapping* `cellina.Cellina` over
   reimplementing the VAE. Keep the same counterfactual API
   (`get_counterfactual_expression`) so it drops into the existing eval harness.
4. **Evaluate** by feeding your model's predictions through `scripts/eval.py`
   (`make_comparison_adata` + `run_loo_eval`), exactly as the notebook does for cellina
   and CPA.

## Conventions to match

- Inputs are **count-like** AnnData; raw counts live in `adata.layers['counts']`.
- Cell-type labels: `adata.obs['coarse_type']`; spatial domain: `adata.obs['typ_clean']`
  (values `REF` / `TVA` / `CRC`); control domain is `REF`, held-out domain is `CRC`.
- Metrics come from `scripts/metrics.py::compare_adatas`.
