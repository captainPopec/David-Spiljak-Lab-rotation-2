# wsi_cellina

Bringing **whole-slide-image (WSI)** information into the
[`cellina`](../cellina) spatial-perturbation framework, and benchmarking it against
existing perturbation models.

This repo is a clean starting point for a student project. It has two halves:

1. **A perturbation benchmark.** A leave-one-out (LOO) harness that trains `cellina` and
   the **CPA** competitor on spatial CRC data, holds out one cell type / region, and
   scores each model's predicted perturbation response with a shared metrics module.
2. **WSI groundwork.** Scaffolding (an empty `src/wsi` package + two skeleton notebooks)
   for the new method: fusing WSI image features (via [LazySlide](https://lazyslide.readthedocs.io/)
   / [SpatialData](https://spatialdata.scverse.org/)) into the `cellina` spatial context.

## Repository structure

```
wsi_cellina/
├── README.md
├── pyproject.toml                # makes `wsi` importable: `pip install -e .`
├── envs/                         # conda environments (see below)
│   ├── cellina.yml
│   ├── cpa_env.yml
│   └── wsi.yml
├── src/
│   └── wsi/                      # <- the NEW method goes here (your work)
│       ├── __init__.py
│       ├── model.py              # WSICellina scaffold + TODOs
│       └── README.md             # the plan for this package
├── scripts/                      # evaluation harness + competitor baselines
│   ├── configs/                  # CRC-only model + data configs
│   │   ├── adata_crc_config.py
│   │   ├── cellina_config.py
│   │   └── cpa_config.py
│   ├── metrics.py                # compare_adatas / get_perturbation_effects (the metrics)
│   ├── eval.py                   # CRC preprocessing, LOO splits, metric glue
│   ├── cpa_runner.py             # CPA train + counterfactual (the baseline)
│   └── utils.py                  # set_seed
└── notebooks/
    ├── loo_eval.ipynb            # cellina vs CPA, leave-one-cell-type-out (Myeloid, crc_232)
    ├── 01_spatialdata_load.ipynb # skeleton: load CRC as SpatialData
    └── 02_lazyslide.ipynb        # skeleton: WSI features with LazySlide
```

`scripts/` holds the harness and **competitors** (cellina is consumed from its own
package; CPA wrapping lives in `cpa_runner.py`). `src/` holds the **new methods** you
develop. The metrics in `scripts/metrics.py` are the single source of truth for scoring.

> Note: the CPA module is `cpa_runner.py`, not `cpa.py`, on purpose — a file named
> `cpa.py` on the path would shadow the `cpa` package it imports.

## Environments

`cellina` and `cpa` depend on **incompatible** `scvi-tools` versions, so they get
separate envs. Create whichever you need:

```bash
conda env create -f envs/cellina.yml   # cellina training + all evaluation/plotting
conda env create -f envs/cpa_env.yml    # the CPA baseline (cpa-tools)
conda env create -f envs/wsi.yml        # SpatialData + LazySlide notebooks (starter stub)
```

| Env | Use for |
|---|---|
| `cellina` | `loo_eval.ipynb` section A + C, and anything calling `cellina` / the metrics |
| `cpa` | `loo_eval.ipynb` section B (the CPA baseline) |
| `wsi` | `01_spatialdata_load.ipynb`, `02_lazyslide.ipynb` |

Install this repo's package into the active env so `import wsi` works:

```bash
pip install -e .
```

`cellina` itself is an external dependency — it is pinned in `envs/cellina.yml`; for local
development install it editable from the sibling checkout: `pip install -e ../cellina`.

## Data

Scripts and notebooks resolve data under a `DATA_ROOT` environment variable (default `.`).
The benchmark uses CRC CosMx sample `crc_232.h5ad`, expected at
`$DATA_ROOT/datasets/crc/crc_232.h5ad`. If absent, `loo_eval.ipynb` falls back to the
public Zenodo copy via `sc.read(..., backup_url=...)`.

Required `.obs` columns (the rest is derived by `eval.preprocess_crc`): `ist` (fine cell
type → `coarse_type`), `typ` (raw region → `typ_clean` ∈ {REF, TVA, CRC}), and the spatial
coordinate columns `CenterX_global_px` / `CenterY_global_px`. Control region is `REF`;
the held-out region is `CRC`.

## Running the benchmark

`notebooks/loo_eval.ipynb` reproduces the cellina-vs-CPA comparison on a single held-out
cell type (Myeloid) — an extension of the [`cellina` tutorial](../cellina/docs/tutorial.ipynb).
Because of the env split it runs in three sections:

1. **A — `cellina` kernel:** preprocess → LOO split → train cellina → counterfactual;
   writes `notebooks/preds/observed.h5ad` and `cellina_pred.h5ad`.
2. **B — `cpa` kernel:** train CPA on the same split → counterfactual; writes
   `cpa_pred.h5ad`.
3. **C — either kernel:** load the three `.h5ad`s, score with
   `eval.run_loo_eval` (`compare_adatas`), and plot observed-vs-predicted logFC.

The same harness extends to other cell types (change `HOLDOUT_CT`) and to
leave-one-region-out (`split_indices(..., holdout_full_domain=True)`).

## For the student

- Read `src/wsi/README.md` — it lays out the plan for the new method.
- Get the data flowing through `01_spatialdata_load.ipynb` and `02_lazyslide.ipynb`.
- Establish the cellina/CPA baseline with `loo_eval.ipynb`.
- Implement `wsi.model.WSICellina` (wrap `cellina.Cellina`; keep its counterfactual API),
  then evaluate it with the *same* `eval.make_comparison_adata` + `eval.run_loo_eval`
  path the notebook uses for cellina and CPA.
