"""
Fold every ROI artifact that currently lives outside the zarr into a single self-contained
zarr: OVCA_with_embeddings.zarr.

What is being consolidated (all previously loose files under data/ovca_roi_embeddings*/):
  - ovca_roi_cell_embeddings.h5ad         (uni2, 1536-d)     -> tables/roi_cells obsm["uni2"]
  - ..._virchow2_backup/...h5ad           (virchow2, 2560-d)  -> tables/roi_cells obsm["virchow2"]
    NOTE: the "_backup" dir is NOT a backup -- it is the only copy of the virchow2 embeddings.
    The extract script overwrites its output h5ad per run, so the virchow2 run was moved aside
    to survive the uni2 run.
  - cell_type_{uni2,virchow2}_predictions.csv                 -> tables/roi_cells obs
  - domain_{uni2,virchow2}_predictions.csv                    -> tables/roi_cells obs
    (the unprefixed cell_type_/domain_predictions.csv are byte-identical duplicates of the
     uni2 files -- verified by md5 -- so they are skipped.)
  - ovca_roi_definitions.geojson          (36 ROI circles)    -> shapes/roi_polygons
  - ovca_roi_manifest.json                (run config)        -> tables/roi_cells uns["roi_manifest"]

The ROI cells are a 17,278 / 407,124 (4.2%) subset of the main table, so they go in as their
own region-table rather than as NaN-padded obsm on the main table (that would cost ~6.7 GB of
mostly-NaN). The new table carries its own copy of the gene expression for those cells, so it
is a complete AnnData on its own: X = counts, obsm = embeddings, obs = labels + predictions.

The main `table` element and all images/labels/points are left untouched.
"""

import json
from pathlib import Path

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity

# The ROI obs index and its string columns come through as pandas nullable string arrays.
# anndata refuses to write those unless opted in. The main `table` in this same zarr is already
# stored with StringDType, so this matches what is on disk rather than introducing a new encoding.
ad.settings.allow_write_nullable_strings = True

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
UNI2_DIR = DATA / "ovca_roi_embeddings"
VIRCHOW2_DIR = DATA / "ovca_roi_embeddings_virchow2_backup"
DEST = Path("/scratch/spiljak/wsi_ood_datasets/OVCA_with_embeddings.zarr")

TABLE_NAME = "roi_cells"
SHAPES_NAME = "roi_polygons"


def log(msg):
    print(f"[integrate] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load the two embedding h5ads. Same 17,278 cells, same row order (verified), different model.
# ---------------------------------------------------------------------------
uni2 = ad.read_h5ad(UNI2_DIR / "ovca_roi_cell_embeddings.h5ad")
virchow2 = ad.read_h5ad(VIRCHOW2_DIR / "ovca_roi_cell_embeddings.h5ad")

if not (uni2.obs_names == virchow2.obs_names).all():
    raise SystemExit("uni2 and virchow2 h5ads disagree on cell order; refusing to merge blind")
log(f"loaded embeddings: uni2 {uni2.obsm['uni2'].shape}, virchow2 {virchow2.obsm['virchow2'].shape}")

obs = uni2.obs.copy()
cell_ids = obs["cell_id"].astype(str).to_numpy()

# ---------------------------------------------------------------------------
# 2. Merge the logreg predictions on cell_id. These cover slightly fewer cells than the
#    embeddings (cell_type 17,265; domain 14,170), so unmatched cells get NaN -- expected.
# ---------------------------------------------------------------------------
for task, csv_name in [
    ("cell_type", "cell_type_{}_predictions.csv"),
    ("domain", "domain_{}_predictions.csv"),
]:
    for model in ["uni2", "virchow2"]:
        path = UNI2_DIR / csv_name.format(model)
        pred = pd.read_csv(path).set_index("cell_id")
        # true_* and roi_id already exist in obs from the h5ad; keep only what's new.
        keep = [c for c in pred.columns if c.startswith("pred_") or c.startswith("proba_")]
        keep += ["split"]
        pred = pred[keep]

        def rename(c):
            # pred_/proba_ columns already name the task, so only the model needs appending;
            # "split" does not, and differs per task. Domain class names contain spaces
            # ("Fallopian tube"), which spatialdata rejects as obs keys.
            if c == "split":
                out = f"split_{task}_{model}"
            elif c.startswith("pred_"):
                out = f"{c}_{model}"
            elif c.startswith("proba_"):
                out = f"proba_{task}_{c[len('proba_'):]}_{model}"
            else:
                out = c
            return out.replace(" ", "_")

        pred.columns = [rename(c) for c in pred.columns]
        # obs is already indexed by cell_id (and keeps it as a column too), so join on the index.
        obs = obs.join(pred)
        log(f"merged {path.name}: {len(pred)} rows -> {len(keep)} cols")

# ---------------------------------------------------------------------------
# 3. Open the destination zarr and pull the pieces the new table needs from the main table:
#    the integer cell_labels instance key (spatialdata annotates labels by int, not by
#    string cell_id) and the gene expression for these cells.
# ---------------------------------------------------------------------------
log(f"reading {DEST}")
sdata = sd.read_zarr(DEST)
main = sdata["table"]

main_ids = main.obs["cell_id"].astype(str).to_numpy()
pos = pd.Series(np.arange(len(main_ids)), index=main_ids)
missing = ~np.isin(cell_ids, main_ids)
if missing.any():
    raise SystemExit(f"{missing.sum()} ROI cells absent from main table; refusing to merge")
rows = pos.loc[cell_ids].to_numpy()

sub = main[rows].copy()  # gene expression + var for exactly the ROI cells, in ROI order

# Carry over the instance/region keys so the table is a valid annotation of cell_labels.
obs["cell_labels"] = sub.obs["cell_labels"].to_numpy()
obs["region"] = sub.obs["region"].to_numpy()
obs["region"] = obs["region"].astype("category")

# ---------------------------------------------------------------------------
# 4. Assemble the ROI table: counts in X, both embeddings in obsm, spatial coords carried over.
# ---------------------------------------------------------------------------
roi = ad.AnnData(
    X=sub.X.copy(),
    obs=obs.reset_index(drop=True),
    var=sub.var.copy(),
)
roi.obsm["uni2"] = np.asarray(uni2.obsm["uni2"])
roi.obsm["virchow2"] = np.asarray(virchow2.obsm["virchow2"])
roi.obsm["spatial"] = np.asarray(sub.obsm["spatial"])
roi.obs_names = cell_ids

with open(UNI2_DIR / "ovca_roi_manifest.json") as f:
    roi.uns["roi_manifest"] = json.load(f)
roi.uns["provenance"] = {
    "built_by": "scripts/integrate_roi_into_zarr.py",
    "source_zarr": "OVCA.zarr",
    "uni2_h5ad": str(UNI2_DIR / "ovca_roi_cell_embeddings.h5ad"),
    "virchow2_h5ad": str(VIRCHOW2_DIR / "ovca_roi_cell_embeddings.h5ad"),
}

roi = TableModel.parse(
    roi,
    region="cell_labels",
    region_key="region",
    instance_key="cell_labels",
)

# ---------------------------------------------------------------------------
# 5. ROI polygons. Geometry is already in global px (see extract_cell_embeddings_roi.py
#    section 8), so it attaches to the "global" coordinate system with Identity.
# ---------------------------------------------------------------------------
gdf = gpd.read_file(UNI2_DIR / "ovca_roi_definitions.geojson")
shapes = ShapesModel.parse(gdf, transformations={"global": Identity()})
log(f"parsed {len(gdf)} ROI polygons")

# ---------------------------------------------------------------------------
# 6. Write the two new elements. Additive: nothing existing is touched.
# ---------------------------------------------------------------------------
sdata[TABLE_NAME] = roi
sdata[SHAPES_NAME] = shapes
sdata.write_element(TABLE_NAME)
log(f"wrote tables/{TABLE_NAME}: {roi.shape}, obsm={ {k: v.shape for k, v in roi.obsm.items()} }")
sdata.write_element(SHAPES_NAME)
log(f"wrote shapes/{SHAPES_NAME}")
log("done")
