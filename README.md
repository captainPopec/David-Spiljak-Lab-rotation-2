# wsi_ood: H&E embeddings as a microenvironment representation for counterfactual gene expression

## About

[Cellina](https://doi.org/10.48550/arXiv.2606.08493) splits a cell's gene expression into an
intrinsic part (who the cell is) and an extrinsic part (what its neighbourhood does to it), which
lets you ask counterfactual questions such as *"what would this cell look like in a tumour?"*.
Cellina learns the neighbourhood from spatial transcriptomics (ST), which is expensive. H&E images
are cheap and routine.

This repository tests whether **embeddings from pathology foundation models (UNIv2, Virchow2, …)
can replace the ST-derived microenvironment representation**, so that such counterfactuals could
be posed from H&E alone. Using a 10x Xenium Prime ovarian cancer (OVCA) section with a matched H&E
image, it:

1. benchmarks how well per-cell H&E tile embeddings recover tissue domain and cell type;
2. trains a decoder-only analogue of Cellina that predicts expression from
   `[cell-type one-hot | context]`, where the context is either a domain one-hot (**labels model**)
   or an H&E embedding (**embeddings model**);
3. moves Stromal cells *in silico* from Smooth muscle into Tumor and scores the prediction
   against held-out Tumor regions, with [Path2Space](https://doi.org/10.1101/2024.10.16.618609)
   as an external H&E-to-expression baseline.

The project was done as a lab rotation at EMBL and will be continued as a Master's thesis.

---

## Repository structure

```
main_wsi_ood/
├── src/wsi/                  # installable package: the models
├── scripts/                  # data building, embedding extraction, evaluation harness
│   ├── configs/              # hyperparameters and dataset settings
│   └── preprocess/           # raw 10x bundles -> SpatialData .zarr
├── notebooks/                # experiments and figures
├── envs/                     # conda environment files
├── data/                     # all inputs/outputs (git-ignored, not in the repo)
└── pyproject.toml            # `pip install -e .` makes `wsi` importable
```

### `src/wsi/`: the models

| File | What it does |
|---|---|
| `labels_baseline.py` | `LabelsBaseline`: one-hot cell type + one-hot domain → scvi-tools `DecoderSCVI` → negative-binomial counts. No encoder. The counterfactual just swaps the domain label. |
| `embeddings_baseline.py` | `EmbeddingsBaseline`: the same model, but the domain one-hot is replaced by a per-cell H&E embedding. The counterfactual substitutes the embedding of a randomly drawn *donor* Tumor cell. Embeddings must be standardised by the caller (on training cells only). |
| `model.py` | Empty scaffold for the full WSI + Cellina model (future work). |

Both models expose the same API as `cellina.Cellina` (`setup_anndata`, `train`,
`get_normalized_expression`, `get_counterfactual_expression`), so they plug into the same evaluation harness.

### `scripts/`: pipeline and evaluation

| File | What it does |
|---|---|
| `preprocess/build_ovca.py` | Builds `data/OVCA.zarr` (SpatialData) from the raw Xenium bundle, the registered H&E and the SpatialFusion cell-type/domain annotations. `build_prad.py` and `build_brca.py` do the same for two other datasets; `DATASETS.md` explains why these three were chosen. |
| `preprocess/xenium_zarr_compat.py` | Makes newer (XOA ≥ 3.0) zipped Xenium bundles readable by `spatialdata_io`. |
| `extract_cell_embeddings_roi.py` | Places 36 non-overlapping circular ROIs (r = 150 µm) across the 5 tissue domains and extracts one 112 µm H&E tile per cell with LazySlide, embedding it with a foundation model (`MODEL_KEYS`). ROIs are spaced so that tiles from different ROIs never share pixels, which makes an ROI-level train/test split leakage-free. `_test3pct.py` is the same on a 3% subsample for quick checks. |
| `submit_extract_cell_embeddings_roi.sh` | SLURM submission for the script above (GPU, `lazyslide-env`). |
| `integrate_roi_into_zarr.py` | Folds ROI embeddings, ROI polygons and classifier predictions into one self-contained `OVCA_with_embeddings.zarr`. |
| `prepare_ovca_loo_data.py` | Exports the ~14k labelled ROI cells (counts, labels, ROI ids, UNI2/Virchow2 embeddings) to a plain `.npz`/`.csv`/`.npy` bundle in `data/ovca_loo/`. It exists because the zarr (v3) cannot be read by the older zarr/anndata in the training env. |
| `eval.py` | Leave-one-out (LOO) harness: preprocessing, train/val/test splits (cell-type, domain and ROI-level hold-outs) and glue to the metrics. |
| `metrics.py` | Population-level metrics: Pearson r and RMSE of logFC, Spearman of mean expression, precision@k, energy distance in PCA space (`compare_adatas`). |
| `input_influence_ablation.py` | Decodes a full cell-type × context grid from the saved models and splits output variance (two-way ANOVA) into cell-type vs context contributions. |
| `cpa_runner.py` | CPA competitor baseline (from the earlier CRC benchmark). |
| `utils.py` | Seeding helper. |
| `configs/` | `adata_ovca_config.py` (OVCA task: Smooth muscle → Tumor, held-out cell type Stromal, held-out ROIs 3/6/8/9), `labels_config.py` / `embeddings_config.py` (identical training settings by design), `cellina_config.py`, `cpa_config.py`, `adata_crc_config.py`. |

### `notebooks/`

Main OVCA analysis, in the order it is run:

| Notebook | What it does |
|---|---|
| `04_explore_datasets.ipynb` | Loads the built `.zarr` datasets and plots annotations and a UMAP. |
| `05_roi_embedding_classifiers.ipynb` | Logistic regression on tile embeddings to predict domain and cell type, with an ROI-level split (embedding benchmark). |
| `gene_expression_prediction.ipynb` | Runs Path2Space (CTransPath features, Macenko-normalised tiles) on the same ROI cells. |
| `OVCA_03_loo.ipynb` | Trains the labels and embeddings models, runs the Smooth muscle → Tumor counterfactual and scores it (OOD = held-out Stromal, ID = other cell types). |
| `OVCA_03_loo_with_baseline.ipynb` | Same experiment with Path2Space added as an external baseline; produces the final metric tables. |
| `plotting_results.ipynb` | Figures from the cached predictions and metrics (no training). |

Earlier / exploratory:

| Notebook | What it does |
|---|---|
| `03_loo_eval.ipynb` | Original CRC (CosMx) LOO benchmark: Cellina vs labels baseline, held-out Myeloid. |
| `03_loo_eval_path2space.ipynb` | First port of that benchmark to OVCA, with Cellina. |
| `01_spatialdata_load.ipynb`, `02_lazyslide.ipynb` | Starter skeletons from the project setup. |

---

## Getting started

```bash
# 1. environment (see note below), then make the package importable
pip install -e .

# 2. build the dataset            (env: wsi / lazyslide-env)
python scripts/preprocess/build_ovca.py

# 3. per-cell H&E embeddings, run once per encoder via MODEL_KEYS  (GPU, lazyslide-env)
sbatch scripts/submit_extract_cell_embeddings_roi.sh

# 4. classifier benchmark          -> notebooks/05_roi_embedding_classifiers.ipynb
# 5. consolidate + export          (lazyslide-env)
python scripts/integrate_roi_into_zarr.py
python scripts/prepare_ovca_loo_data.py
# 6. Path2Space baseline           -> notebooks/gene_expression_prediction.ipynb
# 7. train, counterfactual, score  -> notebooks/OVCA_03_loo_with_baseline.ipynb   (cellina_edge)
# 8. figures                       -> notebooks/plotting_results.ipynb
```

**Environments.** `envs/` holds `wsi.yml` (SpatialData/LazySlide), `cellina.yml` and
`cpa_env.yml`. The OVCA runs themselves used two local environments, `lazyslide-env` (image
processing, zarr v3) and `cellina_edge` (scvi-tools + Cellina, model training), which are not yet
exported to `envs/`.

**Data.** Everything under `data/` is git-ignored. The raw slide is the public
[10x Xenium Prime FFPE Human Ovarian Cancer](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer)
dataset; cell-type and domain annotations come from
[SpatialFusion](https://doi.org/10.64898/2026.03.16.712056).
