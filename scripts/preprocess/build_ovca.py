"""Build OVCA.zarr SpatialData for the Xenium Prime Ovarian Cancer FFPE slide.

Reads the raw 10x Xenium bundle + post-Xenium H&E + the SpatialFusion author
annotations and writes a SpatialData object bundling:
  - images: `he` (registered via the 10x alignment matrix) + `morphology_focus`
  - labels/shapes: native cell + nucleus segmentation  (R4)
  - table: cell-feature matrix with `cell_type`, `cell_type_minor`, `domain` (R2/R3)
  - points: `domain_points` (cell centroids carrying domain + cell_type)

Run inside the `wsi` env. The pip scipy needs the env's newer libstdc++ on the
loader path, so invoke as either:
    conda activate wsi && python scripts/preprocess/build_ovca.py
or:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/preprocess/build_ovca.py

Expects the raw inputs already staged under notebooks/datasets/_raw/OVCA/ by the
download step (bundle/, he_image.ome.tif, he_imagealignment.csv, celltypes_ovca.csv,
region_annotations_ovca.csv). See notebooks/datasets/DATASETS.md for provenance.
"""
import os
import pandas as pd

import spatialdata as sd  # noqa: F401
import spatialdata_io
from spatialdata.transformations import get_transformation

# --- paths (repo-relative so the script is location-independent) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASETS = os.path.join(REPO_ROOT, "notebooks", "datasets")
RAW = os.environ.get("OVCA_RAW", os.path.join(DATASETS, "_raw", "OVCA"))
OUT = os.environ.get("OVCA_OUT", os.path.join(DATASETS, "OVCA.zarr"))

BUNDLE = os.path.join(RAW, "bundle")
HE_TIF = os.path.join(RAW, "he_image.ome.tif")
ALIGN = os.path.join(RAW, "he_imagealignment.csv")
CELLTYPES = os.path.join(RAW, "celltypes_ovca.csv")
REGIONS = os.path.join(RAW, "region_annotations_ovca.csv")


def find_xenium_root(base):
    for root, _dirs, files in os.walk(base):
        if "experiment.xenium" in files:
            return root
    raise FileNotFoundError("experiment.xenium not found under " + base)


def main():
    xen_root = find_xenium_root(BUNDLE)
    print("Xenium root:", xen_root, flush=True)

    print("=== Reading Xenium bundle (transcripts EXCLUDED) ===", flush=True)
    sdata = spatialdata_io.xenium(
        xen_root,
        cells_boundaries=True,
        nucleus_boundaries=True,
        cells_as_circles=False,
        cells_labels=True,
        nucleus_labels=True,
        transcripts=False,        # skip huge transcripts table to save space
        morphology_mip=True,
        morphology_focus=True,
        aligned_images=False,     # we add H&E explicitly below
        cells_table=True,
    )
    print(sdata, flush=True)

    # === Add H&E image, registered via the 10x alignment matrix ===
    print("=== Reading H&E ome.tif with alignment file ===", flush=True)
    he = spatialdata_io.xenium_aligned_image(
        image_path=HE_TIF,
        alignment_file=ALIGN,
        rgba=False,
    )
    sdata.images["he"] = he
    print("H&E added as 'he'.", flush=True)
    try:
        print("H&E transformations:", get_transformation(sdata.images["he"], get_all=True), flush=True)
    except Exception as e:
        print("  (could not fetch transformation:", e, ")", flush=True)

    # === Identify the table ===
    tname = list(sdata.tables)[0]
    table = sdata.tables[tname]
    print("Table name:", tname, "shape:", table.shape, flush=True)
    print("obs columns:", list(table.obs.columns), flush=True)

    obs = table.obs
    cell_ids = obs["cell_id"].astype(str) if "cell_id" in obs.columns else obs.index.astype(str)
    print("Example obs cell ids:", list(cell_ids[:3]), flush=True)

    # === Join author annotations ===
    ct = pd.read_csv(CELLTYPES).set_index("cell_id")
    rg = pd.read_csv(REGIONS).set_index("cell_id")
    print("celltypes cols:", list(ct.columns), "n=", len(ct), flush=True)
    print("regions cols:", list(rg.columns), "n=", len(rg), flush=True)

    n = len(cell_ids)
    cell_type = cell_ids.map(ct["major_celltype"])
    minor = cell_ids.map(ct["minor_celltype"]) if "minor_celltype" in ct.columns else None
    domain = cell_ids.map(rg["Region_annotation"])

    ct_match = int(cell_type.notna().sum())
    dm_match = int(domain.notna().sum())
    print(f"cell_type match: {ct_match}/{n} = {100*ct_match/n:.2f}%", flush=True)
    print(f"domain match: {dm_match}/{n} = {100*dm_match/n:.2f}%", flush=True)

    # fall back to prefix-stripped ids (e.g. 'TENXxxx_aaaa-1' -> 'aaaa-1') if needed
    if ct_match / n < 0.5:
        stripped = cell_ids.str.replace(r"^.*_", "", regex=True)
        cm2 = stripped.map(ct["major_celltype"])
        if int(cm2.notna().sum()) > ct_match:
            print("Using prefix-stripped ids for matching", flush=True)
            cell_type = cm2
            minor = stripped.map(ct["minor_celltype"]) if "minor_celltype" in ct.columns else None
            domain = stripped.map(rg["Region_annotation"])

    obs["cell_type"] = pd.Categorical(cell_type.values)
    if minor is not None:
        obs["cell_type_minor"] = pd.Categorical(minor.values)
    obs["domain"] = pd.Categorical(domain.values)
    table.obs = obs

    print("=== value_counts cell_type ===", flush=True)
    print(obs["cell_type"].value_counts(dropna=False), flush=True)
    print("=== value_counts domain ===", flush=True)
    print(obs["domain"].value_counts(dropna=False), flush=True)

    # === R3: domain also as a points element (cell centroids carrying domain) ===
    print("=== Building domain points element ===", flush=True)
    try:
        from spatialdata.models import PointsModel
        xcol = next((c for c in ["x_centroid", "x_centroid_px", "cell_centroid_x"] if c in obs.columns), None)
        ycol = next((c for c in ["y_centroid", "y_centroid_px", "cell_centroid_y"] if c in obs.columns), None)
        if xcol and ycol:
            pdf = pd.DataFrame({
                "x": obs[xcol].values.astype(float),
                "y": obs[ycol].values.astype(float),
                "domain": obs["domain"].astype(str).values,
                "cell_type": obs["cell_type"].astype(str).values,
            })
            pts = PointsModel.parse(pdf, coordinates={"x": "x", "y": "y"}, feature_key="domain")
            sdata.points["domain_points"] = pts
            print("Added domain_points from obs centroids (cols %s,%s)" % (xcol, ycol), flush=True)
        else:
            cs_name = next((s for s in sdata.shapes if "cell" in s and "nucle" not in s), list(sdata.shapes)[0])
            shp = sdata.shapes[cs_name]
            cents = shp.geometry.centroid
            pdf = pd.DataFrame({"x": cents.x.values, "y": cents.y.values}, index=shp.index)
            pdf["domain"] = pdf.index.map(pd.Series(obs["domain"].astype(str).values, index=obs.index))
            pdf["cell_type"] = pdf.index.map(pd.Series(obs["cell_type"].astype(str).values, index=obs.index))
            tr = get_transformation(shp, get_all=True)
            pts = PointsModel.parse(pdf.reset_index(drop=True), coordinates={"x": "x", "y": "y"},
                                    feature_key="domain", transformations=tr)
            sdata.points["domain_points"] = pts
            print(f"Added domain_points from {cs_name} centroids", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("WARN: could not build domain points:", repr(e), flush=True)

    # === Write ===
    print("=== Writing zarr ===", flush=True)
    if os.path.exists(OUT):
        import shutil
        shutil.rmtree(OUT)
    sdata.write(OUT)
    print("WROTE", OUT, flush=True)
    print("DONE_BUILD", flush=True)


if __name__ == "__main__":
    main()
