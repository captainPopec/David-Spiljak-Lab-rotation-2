#!/usr/bin/env python
"""Build BRCA.zarr: Xenium read + H&E + author cell types + pathologist regions.

Dataset: Xenium_FFPE_Human_Breast_Cancer_Rep1 (Janesick et al. 2023, Nat Commun),
313-gene breast panel (280 panel + 33 custom), HEST sample NCBI875.

The author cell_id is a POSITIONAL index into the bundle's native cell order (HEST QC
subset), so we join on position, not barcode. No region geojson is published — domains
were retraced from the Nat Commun figure as per-cell labels, so `pathology_regions` is
derived by dissolving the native cell-boundary polygons per domain (cleaned with
make_valid/buffer(0) to avoid GEOS TopologyExceptions).

Run in the `wsi` env (`conda activate wsi`, or prefix `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`).
Expects raw inputs staged under notebooks/datasets/_raw/BRCA/ (bundle/outs/,
he_image.ome.tif, he_imagealignment.csv, celltypes_brca.csv,
breast_ffpe_region_annotations.csv). See DATASETS.md.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
warnings.filterwarnings("ignore")

import spatialdata as sd
import spatialdata_io
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import Identity
from spatialdata_io import xenium_aligned_image

# --- paths (repo-relative) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS = os.path.join(REPO_ROOT, "notebooks", "datasets")
RAW = os.environ.get("BRCA_RAW", os.path.join(DATASETS, "_raw", "BRCA"))
OUT = os.environ.get("BRCA_OUT", os.path.join(DATASETS, "BRCA.zarr"))
BUNDLE = os.path.join(RAW, "bundle", "outs")
HE_TIF = os.path.join(RAW, "he_image.ome.tif")
ALIGN_CSV = os.path.join(RAW, "he_imagealignment.csv")
CELLTYPES = os.path.join(RAW, "celltypes_brca.csv")
REGIONS = os.path.join(RAW, "breast_ffpe_region_annotations.csv")


def main():
    print("=== Reading Xenium bundle (transcripts skipped) ===", flush=True)
    try:
        sdata = spatialdata_io.xenium(
            BUNDLE, cells_boundaries=True, nucleus_boundaries=True,
            transcripts=False, cells_as_circles=False,
        )
    except TypeError:
        sdata = spatialdata_io.xenium(BUNDLE, transcripts=False)
    print(sdata, flush=True)

    tkey = "table" if "table" in sdata.tables else list(sdata.tables.keys())[0]
    table = sdata.tables[tkey]
    n_cells, n_genes = table.n_obs, table.n_vars
    print(f"n_cells={n_cells} n_genes={n_genes}", flush=True)

    # ---- Author annotations (positional cell_id join) ----
    ct = pd.read_csv(CELLTYPES, index_col=0)
    rg = pd.read_csv(REGIONS, index_col=0)
    print("celltypes columns:", ct.columns.tolist(), "n=", len(ct), flush=True)
    print("regions columns:", rg.columns.tolist(), "n=", len(rg), flush=True)

    table.obs = table.obs.copy()
    table.obs["_pos"] = np.arange(n_cells)
    ct_idx = ct.set_index("cell_id")
    match0 = table.obs["_pos"].isin(ct_idx.index).sum()
    match1 = table.obs["_pos"].add(1).isin(ct_idx.index).sum()
    print(f"match 0-based={match0} 1-based={match1}", flush=True)
    key = (table.obs["_pos"] + 1) if match1 > match0 else table.obs["_pos"]
    print("using", "1-based" if match1 > match0 else "0-based", "positional index", flush=True)

    table.obs["cell_type"] = key.map(ct_idx["major_celltype"]).astype("category")
    table.obs["cell_subtype"] = key.map(ct_idx["minor_celltype"]).astype("category")
    table.obs["domain"] = key.map(rg["Region_annotation"]).astype("category")

    n_ct = int(table.obs["cell_type"].notna().sum())
    n_dom = int(table.obs["domain"].notna().sum())
    print(f"cell_type matched: {n_ct}/{n_cells} = {100*n_ct/n_cells:.1f}%", flush=True)
    print(f"domain matched: {n_dom}/{n_cells} = {100*n_dom/n_cells:.1f}%", flush=True)
    print("cell_type value_counts:\n", table.obs["cell_type"].value_counts(dropna=False), flush=True)
    print("domain value_counts:\n", table.obs["domain"].value_counts(dropna=False), flush=True)
    if n_ct / n_cells < 0.5:
        print("!!! LOW MATCH RATE -- aborting, ids do not line up", flush=True)
        sys.exit(2)
    table.obs.drop(columns=["_pos"], inplace=True)

    # ---- H&E image + alignment ----
    print("=== Adding H&E image ===", flush=True)
    aligned = False
    try:
        he = xenium_aligned_image(HE_TIF, ALIGN_CSV)
        aligned = True
        print("H&E loaded with alignment via xenium_aligned_image", flush=True)
    except Exception as e:
        print("xenium_aligned_image failed:", repr(e), flush=True)
        he = None
    if he is None:
        from dask_image.imread import imread as daimread
        img = daimread(HE_TIF).squeeze()
        if img.ndim == 3 and img.shape[0] not in (3, 4) and img.shape[-1] in (3, 4):
            img = np.moveaxis(img, -1, 0)
        he = Image2DModel.parse(img, dims=("c", "y", "x"))
        print("H&E loaded WITHOUT verified alignment (manual read)", flush=True)
    sdata.images["he"] = he
    print("H&E alignment status:", "ALIGNED (10x alignment file applied)" if aligned else "NOT aligned", flush=True)

    # ---- Pathology regions as shapes (dissolve cells per domain) ----
    cell_shapes_key = next((k for k in sdata.shapes if "cell" in k.lower() and "nucle" not in k.lower()),
                           (list(sdata.shapes.keys())[0] if sdata.shapes else None))
    print("cell shapes key for region dissolve:", cell_shapes_key, flush=True)
    if cell_shapes_key is not None:
        cg = sdata.shapes[cell_shapes_key].copy()
        if len(cg) == n_cells:
            cg["domain"] = table.obs["domain"].values
        else:
            dom = pd.Series(table.obs["domain"].values, index=table.obs_names.astype(str))
            cg["domain"] = pd.Index(cg.index.astype(str)).map(dom)
        cg = cg[cg["domain"].notna() & (cg["domain"].astype(str) != "Unassigned")]
        if len(cg) > 0:
            import shapely
            from shapely import make_valid
            cg = cg.copy()
            cg["geometry"] = cg.geometry.apply(make_valid).buffer(0)
            cg = cg[~cg.geometry.is_empty & cg.geometry.notna()]
            doms, geoms = [], []
            for dom_val, grp in cg.groupby("domain"):
                doms.append(str(dom_val))
                geoms.append(shapely.union_all(grp.geometry.values, grid_size=0.01))
            regions = gpd.GeoDataFrame({"domain": doms}, geometry=geoms)
            regions["geometry"] = regions["geometry"].apply(make_valid)
            tr = dict(sdata.shapes[cell_shapes_key].attrs.get("transform", {"global": Identity()}))
            sdata.shapes["pathology_regions"] = ShapesModel.parse(regions, transformations=tr)
            print("Built pathology_regions by dissolving cells per domain; rows:",
                  len(regions), regions["domain"].tolist(), flush=True)

    # ---- Write + validate ----
    print("=== Writing zarr ===", flush=True)
    sdata.write(OUT, overwrite=True)
    print("WROTE", OUT, flush=True)
    print(sd.read_zarr(OUT), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
