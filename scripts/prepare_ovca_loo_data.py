"""Build the OVCA leave-one-out (LOO) input bundle for the LOO notebooks.

WHY THIS SCRIPT EXISTS AT ALL (i.e. why the notebooks can't just read the zarr themselves):
the notebooks run in `cellina_edge` (zarr 2.18.3 / anndata 0.10.8), but `OVCA_with_embeddings.zarr`
was written by `lazyslide-env` (zarr 3.1.6) in **zarr v3 format**, which zarr 2.x cannot read.
So the read has to happen here, in `lazyslide-env`, and the result handed across.

The handoff format is deliberately **not** h5ad. anndata 0.12 writes string columns (and the
obs `_index`) as `ArrowStringArray` -> encoding-type `nullable-string-array`, which anndata
0.10.8 refuses to read ("No read method registered"); and toggling
`allow_write_nullable_strings=True` only makes the *write* succeed, not the read. Verified
empirically both ways. `roi_cells` obs still carries such columns today (e.g. `pred_domain_uni2`
is pandas `string` dtype), so this is not hypothetical. A plain `.npz` + `.csv` + `.npy` bundle
sidesteps the anndata/zarr version skew entirely -- these formats are stable across both envs.

WHAT IT PRODUCES: the ~14.2k ROI cells that have a ground-truth domain, with their real Xenium
counts, micron centroids, WSI-derived domain call, and per-cell H&E foundation-model embeddings:

    data/ovca_loo/counts.npz    csr_matrix [n_cells, 5101] float32, RAW counts
    data/ovca_loo/obs.csv       cell_id, cell_type, domain, pred_domain, roi_id,
                                roi_domain, split_05
    data/ovca_loo/var.csv       gene names (5101)
    data/ovca_loo/spatial.npy   [n_cells, 2] float64 centroids, MICRONS
    data/ovca_loo/uni2.npy      [n_cells, 1536] float32 UNI2 per-cell H&E embeddings
    data/ovca_loo/virchow2.npy  [n_cells, 2560] float32 Virchow2 per-cell H&E embeddings
    data/ovca_loo/manifest.json provenance + the pred-vs-true agreement table

SOURCE: a single table, `OVCA_with_embeddings.zarr/tables/roi_cells` -- the 36 non-overlapping
ROIs from `extract_cell_embeddings_roi.py` (17,278 cells) with counts, spatial, the 05
classifier outputs, and both embeddings already joined onto them. Rows whose ground-truth
`domain` is null are dropped (3,108 of 17,278; the 05 `fit_eval_task` also drops NaN-label rows,
so those cells have no prediction either), which lands the bundle at 14,170.

This supersedes an earlier three-way join (ROI h5ad + `domain_uni2_predictions.csv` +
`OVCA.zarr/tables/table`). That join, its `one_to_one` merge and its drift assert are gone: the
inputs are now one table, so they cannot drift apart. The output is byte-identical to what the
old join produced -- same 14,170 cell_ids in the same order, same counts/spatial/obs -- with the
embeddings added. Verified before the switch.

READ THE CAVEAT IN THE MANIFEST before interpreting anything downstream: `pred_domain` agrees
with `domain` ~99.3% of the time, because 77.5% of these cells sit in ROIs the 05 logistic
regression was *trained* on (100% accuracy there; 96.9% on its held-out ROIs). These are
therefore mostly in-sample predictions, not "domain inferred from H&E" in the deployment sense.
Kept as-is by explicit choice; the notebooks restate this.

Run under `lazyslide-env` (needs zarr 3 to read the zarr):
    /g/stegle/spiljak/programs/miniforge3/envs/lazyslide-env/bin/python \
        scripts/prepare_ovca_loo_data.py
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
# Single source of truth: counts + spatial + 05 classifier outputs + H&E embeddings, one table.
TABLE_PATH = REPO_ROOT / "notebooks" / "datasets" / "OVCA_with_embeddings.zarr" / "tables" / "roi_cells"

OUT_DIR = REPO_ROOT / "data" / "ovca_loo"

# obsm key -> output filename. uni2 is the embedding the 05 headline domain numbers are on (and
# the one every pred_*_uni2 column here comes from); virchow2 rides along so trying the other
# foundation model later is a config change, not a re-extraction.
EMBEDDINGS = {"uni2": "uni2.npy", "virchow2": "virchow2.npy"}

# Columns renamed on the way out. The zarr suffixes the 05 outputs with the embedding they came
# from; the bundle's consumers expect the unsuffixed names. Changing these breaks the notebooks.
RENAME = {"pred_domain_uni2": "pred_domain", "split_domain_uni2": "split_05"}

OBS_COLS = ["cell_id", "cell_type", "domain", "pred_domain", "roi_id", "roi_domain", "split_05"]
# Plain str/object, never pandas' nullable/Arrow string dtypes -- see module docstring.
STR_COLS = ["cell_id", "cell_type", "domain", "pred_domain", "roi_domain", "split_05"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. The one table -------------------------------------------------------------------
    table = ad.read_zarr(TABLE_PATH)
    print(f"[1] {TABLE_PATH.name}: {table.shape}, obsm={sorted(table.obsm)}")

    missing_obsm = set(EMBEDDINGS) - set(table.obsm)
    assert not missing_obsm, f"table is missing expected embeddings: {sorted(missing_obsm)}"

    # --- 2. Keep the cells with a ground-truth domain ----------------------------------------
    # The LOO split and the counterfactual arms are both cut by `domain`, so a cell without one
    # cannot take part in the experiment. These same rows are also the ones 05 scored, so this
    # single filter is what the old three-way inner join used to accomplish.
    keep = table.obs["domain"].notna().to_numpy()
    sub = table[keep].copy()
    print(f"[2] dropped {int((~keep).sum())} cells with no ground-truth domain -> {sub.n_obs} cells")
    assert sub.obs["cell_type"].notna().all(), "domain-labelled cells with a null cell_type"

    # --- 3. obs ------------------------------------------------------------------------------
    obs = sub.obs.reset_index(drop=True).rename(columns=RENAME)[OBS_COLS]
    obs = obs.astype({c: str for c in STR_COLS})
    print(f"[3] obs: {list(obs.columns)}")

    # --- 4. Counts, centroids, genes, embeddings ---------------------------------------------
    # No row mapping needed: X / obsm / obs are the same rows in the same order by construction.
    counts = sp.csr_matrix(sub.X)
    spatial = np.asarray(sub.obsm["spatial"], dtype=np.float64)
    genes = np.asarray(sub.var_names, dtype=object)
    embeddings = {k: np.asarray(sub.obsm[k], dtype=np.float32) for k in EMBEDDINGS}

    # cellina/LabelsBaseline both assume raw counts in the registered layer.
    dense_head = counts[:500].toarray()
    assert np.allclose(dense_head, np.round(dense_head)), "counts are not integer-valued"
    print(f"[4] counts {counts.shape} (nnz={counts.nnz}) | spatial {spatial.shape} (microns)")
    for k, v in embeddings.items():
        assert v.shape[0] == counts.shape[0], f"{k} is not row-aligned with the counts"
        assert np.isfinite(v).all(), f"{k} contains non-finite values"
        print(f"    {k} {v.shape}  ||e|| = {np.linalg.norm(v, axis=1).mean():.2f}"
              f" +- {np.linalg.norm(v, axis=1).std():.2f}")

    # --- 5. Write the version-neutral bundle -------------------------------------------------
    sp.save_npz(OUT_DIR / "counts.npz", counts)
    obs.to_csv(OUT_DIR / "obs.csv", index=False)
    pd.DataFrame({"gene": genes.astype(str)}).to_csv(OUT_DIR / "var.csv", index=False)
    np.save(OUT_DIR / "spatial.npy", spatial)
    for k, fname in EMBEDDINGS.items():
        np.save(OUT_DIR / fname, embeddings[k])

    agreement = float((obs["domain"] == obs["pred_domain"]).mean())
    per_split = (
        obs.assign(agree=obs["domain"] == obs["pred_domain"])
        .groupby("split_05")["agree"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n_cells", "mean": "pred_equals_true"})
    )
    frac_in_sample = float((obs["split_05"] == "train").mean())

    manifest = {
        "n_cells": int(len(obs)),
        "n_genes": int(counts.shape[1]),
        "n_rois": int(obs["roi_id"].nunique()),
        "sources": {"roi_cells_table": str(TABLE_PATH)},
        "spatial_units": "microns (obsm['spatial'] of the roi_cells table, unmodified)",
        "counts": "raw Xenium counts, full 5101-gene panel, no HVG subsetting",
        "embeddings": {
            k: {"file": f, "shape": list(embeddings[k].shape), "dtype": "float32",
                "source": f"obsm['{k}'] of the roi_cells table, unmodified"}
            for k, f in EMBEDDINGS.items()
        },
        "cell_type_counts": obs["cell_type"].value_counts().to_dict(),
        "true_domain_counts": obs["domain"].value_counts().to_dict(),
        "pred_domain_counts": obs["pred_domain"].value_counts().to_dict(),
        "caveat": (
            f"pred_domain agrees with the ground-truth domain for {agreement:.2%} of cells. "
            f"{frac_in_sample:.1%} of these cells lie in ROIs the 05 logistic regression was "
            "TRAINED on (where it is 100% accurate by memorisation); only the split_05=='test' "
            "cells are genuine out-of-sample WSI predictions. Substituting pred_domain for "
            "domain is therefore close to a no-op and does NOT demonstrate domain inference "
            "from H&E alone. Retained as-is by explicit choice."
        ),
        "pred_vs_true_agreement_overall": agreement,
        "pred_vs_true_agreement_by_05_split": per_split.to_dict(orient="index"),
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    print(f"[5] wrote bundle -> {OUT_DIR}")
    print(f"    pred_domain == domain for {agreement:.2%} of cells")
    print(per_split.to_string())


if __name__ == "__main__":
    main()
