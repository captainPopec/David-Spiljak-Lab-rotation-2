#!/usr/bin/env python
"""Build PRAD.zarr: Xenium read + H&E + author cell types + pathologist regions.

Dataset: 10x "Xenium Prime Human Prostate FFPE" (HEST sample TENX157). The
pathologist domain annotation is the only *de novo* board-certified one in the set.

Coordinate systems:
- The SpatialData 'global' system is the Xenium morphology PIXEL grid
  (morphology_focus has an Identity transform). pixel_size = 0.2125 um/px.
- cells.parquet x/y_centroid are in MICRONS -> divide by pixel_size for global pixels.
- The pathologist geojson is drawn in H&E IMAGE PIXEL coords. he_imagealignment.csv maps
  H&E pixels -> global (Xenium pixels); we apply it to the polygons so they live in 'global'.

Run in the `wsi` env (`conda activate wsi`, or prefix `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`).
Expects raw inputs staged under notebooks/datasets/_raw/PRAD/ (outs/, he_image.ome.tif,
he_imagealignment.csv, celltypes_prad.csv, the *_he_image.geojson). See DATASETS.md.
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.affinity import affine_transform
warnings.filterwarnings("ignore")

import spatialdata as sd  # noqa: F401
import spatialdata_io
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity

# --- paths (repo-relative) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS = os.path.join(REPO_ROOT, "notebooks", "datasets")
RAW = os.environ.get("PRAD_RAW", os.path.join(DATASETS, "_raw", "PRAD"))
OUT = os.environ.get("PRAD_OUT", os.path.join(DATASETS, "PRAD.zarr"))
BUNDLE = os.path.join(RAW, "outs")
GEOJSON = os.path.join(RAW, "Xenium_Prime_Human_Prostate_FFPE_he_image.geojson")
CELLTYPES = os.path.join(RAW, "celltypes_prad.csv")
ALIGN = os.path.join(RAW, "he_imagealignment.csv")
PIXEL_SIZE = 0.2125  # um/px, from experiment.xenium


def main():
    print("=== reading Xenium bundle (transcripts skipped) ===", flush=True)
    sdata = spatialdata_io.xenium(BUNDLE, transcripts=False, aligned_images=True, cells_as_circles=False)
    adata = sdata.tables["table"]
    n_cells_total = adata.n_obs
    obs_index = adata.obs_names
    print(f"cells={adata.n_obs} features={adata.n_vars}", flush=True)

    if "feature_types" in adata.var:
        vc = adata.var["feature_types"].value_counts()
        n_genes = int(vc.get("Gene Expression", adata.n_vars))
        print("var feature_types:", vc.to_dict(), flush=True)
    else:
        n_genes = adata.n_vars
    print(f"n_gene_expression_features={n_genes}", flush=True)

    # ---------------- author cell types ----------------
    print("\n=== author cell types ===", flush=True)
    ct = pd.read_csv(CELLTYPES, index_col=0)
    print("celltypes rows:", len(ct), "cols:", list(ct.columns), flush=True)
    ct.index = ct.index.str.replace(r"^TENX157_", "", regex=True)
    cell_type = ct["celltypes"].reindex(obs_index)
    cell_subtype = ct["cellsubtypes"].reindex(obs_index)
    adata.obs["cell_type"] = pd.Categorical(cell_type)
    adata.obs["cell_subtype"] = pd.Categorical(cell_subtype)
    ct_matched = int(cell_type.notna().sum())
    print(f"cell_type matched: {ct_matched}/{n_cells_total} = {100*ct_matched/n_cells_total:.2f}%", flush=True)
    print("cell_type value_counts:\n", adata.obs["cell_type"].value_counts(dropna=False).to_string(), flush=True)

    # ---------------- pathologist regions ----------------
    print("\n=== pathologist geojson regions ===", flush=True)
    gj = json.load(open(GEOJSON))
    feats = gj["features"]
    names = [f.get("properties", {}).get("name") for f in feats]
    geoms_pix = [shape(f["geometry"]) for f in feats]
    print("region names:", names, flush=True)

    align = pd.read_csv(ALIGN, header=None).values  # 3x3 H&E px -> global px
    a, b, xo = align[0]; d, e, yo = align[1]
    shapely_params = [a, b, d, e, xo, yo]   # x' = a*x + b*y + xo ; y' = d*x + e*y + yo
    geoms_global = [affine_transform(g, shapely_params) for g in geoms_pix]
    gdf = gpd.GeoDataFrame({"region": names}, geometry=geoms_global)

    # cell centroids -> global pixel space
    cells_pq = pd.read_parquet(os.path.join(BUNDLE, "cells.parquet")).set_index("cell_id")
    cx = cells_pq["x_centroid"].reindex(obs_index).values / PIXEL_SIZE
    cy = cells_pq["y_centroid"].reindex(obs_index).values / PIXEL_SIZE
    print(f"cell global(px) x:{np.nanmin(cx):.0f}-{np.nanmax(cx):.0f} y:{np.nanmin(cy):.0f}-{np.nanmax(cy):.0f}", flush=True)
    gb = gdf.total_bounds
    print(f"poly global(px) x:{gb[0]:.0f}-{gb[2]:.0f} y:{gb[1]:.0f}-{gb[3]:.0f}", flush=True)

    cells_gdf = gpd.GeoDataFrame({"cell_id": obs_index},
                                 geometry=gpd.points_from_xy(cx, cy), index=obs_index)
    joined = gpd.sjoin(cells_gdf, gdf, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    domain = joined["region"].reindex(obs_index)
    adata.obs["domain"] = pd.Categorical(domain)
    dom_matched = int(domain.notna().sum())
    print(f"domain matched: {dom_matched}/{n_cells_total} = {100*dom_matched/n_cells_total:.2f}%", flush=True)
    print("domain value_counts:\n", adata.obs["domain"].value_counts(dropna=False).to_string(), flush=True)

    # store pathology_regions shapes in 'global' (already global px) -> Identity
    gdf_model = ShapesModel.parse(gdf, transformations={"global": Identity()})
    sdata.shapes["pathology_regions"] = gdf_model

    # ---------------- write ----------------
    print("\n=== writing zarr ===", flush=True)
    print("final obs columns:", list(adata.obs.columns), flush=True)
    if os.path.exists(OUT):
        import shutil
        shutil.rmtree(OUT)
    sdata.write(OUT)
    print("WRITE DONE:", OUT, flush=True)


if __name__ == "__main__":
    main()
