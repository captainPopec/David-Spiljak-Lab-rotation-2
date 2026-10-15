# Pathology foundation model embeddings as a microenvironment representation for counterfactual gene expression prediction



## Repository layout

- `src/wsi/`: the models (labels model and embeddings model), installable as the `wsi` package
- `scripts/`: dataset building, embedding extraction, data export, evaluation harness and metrics
- `scripts/configs/`: task settings and hyperparameters
- `notebooks/`: the experiments and figures
- `envs/`: conda environment files
- `data/`: inputs and outputs (git-ignored)

## Workflow

The analysis runs in the order below. Each step reads the output of the previous one from `data/`.

### 0. Setup

Create the conda environments and make the `wsi` package importable:

```bash
pip install -e .
```

Two environments are used: one for image processing (SpatialData, LazySlide; `envs/wsi.yml`) and one for model training (scvi-tools, Cellina; `envs/cellina.yml`).

### 1. Build the dataset

Download the public [10x Xenium Prime FFPE Human Ovarian Cancer](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer) slide and the [SpatialFusion](https://doi.org/10.64898/2026.03.16.712056) annotations, then combine expression, H&E image, cell types and tissue domains into one SpatialData object:

```bash
python scripts/preprocess/build_ovca.py
```

Optionally, explore the result in `notebooks/04_explore_datasets.ipynb`.

### 2. Extract H&E embeddings

Place 36 non-overlapping regions of interest (ROIs) across the slide and embed a 112 µm H&E tile around every cell with a pathology foundation model. Because the ROIs never share pixels, they can later be split into train and test sets without leakage. Run it once per encoder (set `MODEL_KEYS` in the script, e.g. `virchow2`, then `uni2`). It needs a GPU:

```bash
sbatch scripts/submit_extract_cell_embeddings_roi.sh
```

### 3. Benchmark the embeddings

`notebooks/05_roi_embedding_classifiers.ipynb` trains a logistic regression on the embeddings to predict tissue domain and cell type, using an ROI-level split. This step shows how much of the tissue context each encoder captures.

### 4. Prepare the training data

Merge the embeddings into the dataset, then export the labelled ROI cells into a light bundle that the training environment can read:

```bash
python scripts/integrate_roi_into_zarr.py
python scripts/prepare_ovca_loo_data.py
```

### 5. Run the Path2Space baseline

`notebooks/gene_expression_prediction.ipynb` predicts expression for the same cells with Path2Space, an existing H&E-to-expression model.

### 6. Train and evaluate the counterfactual

`notebooks/OVCA_03_loo_with_baseline.ipynb` does the following:

- trains the labels model and the embeddings models, holding out Stromal cells in Tumor and four whole Tumor ROIs;
- moves Smooth-muscle cells into a Tumor context;
- scores the predictions against the held-out Tumor cells, together with Path2Space.

`OVCA_03_loo.ipynb` is the same experiment without the Path2Space baseline.

### 7. Plot the results

`notebooks/plotting_results.ipynb` makes the figures from the saved predictions and metrics.

Optionally, `scripts/input_influence_ablation.py` measures how much each model's output depends on the cell-type label compared with the context.
