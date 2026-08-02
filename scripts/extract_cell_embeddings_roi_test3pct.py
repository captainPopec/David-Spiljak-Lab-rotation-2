"""Per-cell H&E foundation-model embeddings for the OVCA Xenium Prime slide, extracted from
non-overlapping ROIs so a later ROI-level train/test split has zero pixel overlap between splits.

WHY NOT "one tile per cell, everywhere" (the approach in goofys_wsi_ood/scripts/
OVCA_extract_cell_embeddings*.py): tiles are ~409 H&E-px (112um) wide, cells are packed
every ~7-9um, so adjacent cells' tiles heavily overlap. A random cell-level train/test split
then leaks train pixels into test tiles. This was diagnosed in three throwaway scripts in that
repo (leakage_gradient.py: accuracy decays with distance-to-nearest-train-cell, i.e. memorised,
not learned; domain_leak_demo.py: coordinate-only 1-NN gets high accuracy with zero image info;
OVCA_predict_domains_gridtiles.py: an existing partial fix for a *different* task -- domain
classification via a regular grid -- that sidesteps leakage but drops "one tile per cell").

THE FIX HERE: place circular ROIs (radius 150um) across the slide, at least
`2*R + tile_side*sqrt(2)` apart center-to-center (the sqrt(2) margin makes a EUCLIDEAN gap
sufficient to guarantee no CHEBYSHEV/L-inf overlap between axis-aligned square tiles -- see
the derivation in the module-level comment in `place_rois()` below). Within an accepted ROI,
every cell still gets its own tile (not a grid tile), so per-cell resolution is kept. ROIs are
placed per annotated domain (Tumor / Smooth muscle / Necrosis / Fallopian tube / Ovary) so each
domain gets independent, non-overlapping regions to learn from -- both a cell_type classifier
and a domain classifier can be trained from the same tile set later.

Domain purity ceiling: 90% local purity is NOT achievable for 3 of 5 domains at this radius
(smooth muscle runs in thin bands, necrosis sits in small scattered pockets, fallopian-tube
epithelium is a thin ribbon -- max observed purity 0.78/0.78/0.68 respectively). Purity=0.60 is
used instead; Necrosis (~15 disconnected clusters) and Fallopian tube (one thin ribbon) will
still fall short of the 12-ROI/domain target for structural reasons (verified: not an artifact
of purity threshold or domain processing order -- reducing purity further or reordering domains
does not rescue them, there simply isn't room for more non-overlapping 150um circles).

Coordinate mapping (verified 30/30 against the `cell_labels` raster; 100% of cells land
in-bounds on the H&E):
    centroid (microns, obsm["spatial"])
      --Scale(1/0.2125)-->  global (Xenium morphology px; 0.2125 um/px per experiment.xenium)
      --inv(he affine)-->   H&E pixel space
This build has NO `transcripts` points element (excluded at build time to save space), so
unlike the older goofys_wsi_ood/data/ovca.zarr pipeline there is no Scale transform to read
off an existing element -- 0.2125 is hardcoded from the raw bundle's experiment.xenium.
NOTE: this build's `domain_points` shapes element has a latent bug (Identity transform applied
to the raw micron values, i.e. never scaled to px) -- left alone here, out of scope; this
script does its own correct conversion and does not depend on `domain_points`.

Does NOT train any classifier -- stops at embeddings + ROI metadata. Downstream MLP / logreg
work happens interactively in Jupyter, splitting train/test BY roi_id (obs["roi_id"] in the
output h5ad), never by cell, so no tile from train can share any pixel with a test tile.

Run on a GPU node via scripts/submit_extract_cell_embeddings_roi.sh.
"""

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ad.settings.allow_write_nullable_strings = True
import spatialdata as sd
from spatialdata.transformations import get_transformation
from wsidata import TileSpec, open_wsi
from wsidata.io import add_tiles

from scipy.spatial import cKDTree

import lazyslide as zs

# --- config ---------------------------------------------------------------------
RNG_SEED = 0
# UNI2 dropped for this run to save time after the he-image chunking fix -- run it separately
# later (MODEL_KEYS = ["uni2"]) rather than block this run on both models.
MODEL_KEYS = ["virchow2"]

REPO_ROOT = Path(__file__).resolve().parents[1]
SDATA_PATH = REPO_ROOT / "notebooks" / "datasets" / "OVCA.zarr"
IMAGE_KEY = "he"

# H&E native pixel size (OME PhysicalSizeX/Y agree to 6 decimals), i.e. mpp of he's scale0 grid.
HE_MPP = 0.2738174784591427
# Xenium "global" coordinate system pixel size, from the raw bundle's experiment.xenium
# (`pixel_size: 0.2125`). obsm["spatial"] is in microns; this build has no `transcripts`
# points element to read the Scale transform off, so it is hardcoded here (verified below).
PIXEL_SIZE = 0.2125

TILE_PX = 224
TILE_MPP = 0.5
TILE_SIDE_UM = TILE_PX * TILE_MPP  # 112um

ROI_RADIUS_UM = 150.0
PURITY_THRESHOLD = 0.60
TARGET_ROIS_PER_DOMAIN = 12
# Annotated domains only -- "Unassigned" and NaN are missing labels, not tissue classes.
DOMAINS = ["Tumor", "Smooth muscle", "Necrosis", "Fallopian tube", "Ovary"]
EXCLUDE_LABELS = {"nan", "NaN", "Unassigned"}

# Visualisation only: cap background scatter in the QC plot.
QC_PLOT_MAX_CELLS = 50_000

SUBSAMPLE_FRAC = 0.03  # TEST RUN: keep only ~3% of selected cells, to sanity-check the
# persist()-in-memory fix cheaply before trusting the full-scale run.

out_dir = REPO_ROOT / "data" / "ovca_roi_embeddings_test3pct"  # separate from the real run
out_dir.mkdir(parents=True, exist_ok=True)


def checkpoint(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Load the slide + cell table
# ---------------------------------------------------------------------------
checkpoint(f"Loading spatialdata zarr: {SDATA_PATH}")
sdata = sd.read_zarr(SDATA_PATH)
table = sdata.tables["table"]

# wsidata's SpatialDataImage2DReader.get_region() does `img.sel(...).data.compute()` on EVERY
# tile read, with no caching between calls -- even cells a few px apart (same zarr chunk) each
# independently re-fetch/decompress that chunk from disk. Over thousands of scattered per-cell
# tiles this reads far more off disk than the tile pixels need (measured: ~640GB read for one
# ~17k-tile run, vs. ~9GB of actual tile pixels). The image is only ~4GB -- .persist() forces it
# fully into RAM ONCE, still as a dask-backed array (so wsidata's .compute() calls keep working),
# so every subsequent per-tile read becomes a free in-memory slice instead of a disk hit.
checkpoint("Persisting he image into memory (one-time ~4GB) to eliminate per-tile disk re-reads...")
sdata.images[IMAGE_KEY] = sdata.images[IMAGE_KEY].persist()
checkpoint("Persisted.")

wsi = open_wsi(sdata, image_key=IMAGE_KEY)
wsi.set_mpp(HE_MPP)
checkpoint(f"Loaded OVCA: {table.n_obs} cells, H&E mpp={HE_MPP}")

obs = table.obs
cell_id = obs["cell_id"].astype(str).to_numpy()
domain_all = obs["domain"].astype(str).to_numpy()
cell_type_all = obs["cell_type"].astype(str).to_numpy()
cell_type_minor_all = (
    obs["cell_type_minor"].astype(str).to_numpy() if "cell_type_minor" in obs.columns else None
)
checkpoint("Domain counts:\n" + pd.Series(domain_all).value_counts().to_string())

# ---------------------------------------------------------------------------
# 2. Map all centroids (microns) into H&E pixel space
#    microns --Scale(1/PIXEL_SIZE)--> global (Xenium px) --inv(he affine)--> H&E px
# ---------------------------------------------------------------------------
spatial_um = np.asarray(table.obsm["spatial"], dtype=float)
global_px = spatial_um / PIXEL_SIZE
he_to_global = get_transformation(
    sdata.images[IMAGE_KEY], to_coordinate_system="global"
).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
he_px = (np.c_[global_px, np.ones(len(global_px))] @ np.linalg.inv(he_to_global).T)[:, :2]
he_shape = sdata.images[IMAGE_KEY].shape  # (c, y, x)
in_bounds = (
    (he_px[:, 0] >= 0) & (he_px[:, 0] < he_shape[2]) & (he_px[:, 1] >= 0) & (he_px[:, 1] < he_shape[1])
)
checkpoint(
    f"Centroids -> H&E px: x[{he_px[:,0].min():.0f},{he_px[:,0].max():.0f}] "
    f"y[{he_px[:,1].min():.0f},{he_px[:,1].max():.0f}]  in-bounds={in_bounds.mean():.2%}"
)

# ---------------------------------------------------------------------------
# 3. Place non-overlapping ROIs, stratified per domain
# ---------------------------------------------------------------------------
# Non-overlap rule: two axis-aligned square tiles of side S, centred at points A and B, are
# guaranteed to NOT overlap iff their CHEBYSHEV (L-inf) distance is >= S. Euclidean distance is
# NOT sufficient on its own (two points can be Euclidean-far but Chebyshev-close if offset
# diagonally): worst case, chebyshev(A,B) = euclidean(A,B) / sqrt(2). So requiring
# euclidean(A,B) >= S*sqrt(2) is what actually guarantees chebyshev(A,B) >= S in the worst
# case. For two circular ROIs of radius R, any cell-pair (one per ROI) is at most
# `2R` apart in the worst case beyond the center-to-center distance, so:
#     min_center_dist = 2*R + tile_side*sqrt(2)
# guarantees zero tile overlap between ANY cell in ROI A and ANY cell in ROI B.
MIN_CENTER_DIST_UM = 2 * ROI_RADIUS_UM + TILE_SIDE_UM * np.sqrt(2)
checkpoint(
    f"ROI radius={ROI_RADIUS_UM}um, tile_side={TILE_SIDE_UM}um, "
    f"min ROI center spacing={MIN_CENTER_DIST_UM:.1f}um"
)

tree_all = cKDTree(spatial_um)  # purity checks need ALL cells nearby, any label

roi_records = []  # dicts: roi_id, domain, center_x_um, center_y_um, purity, n_cells (filled later)
accepted_centers_um = []  # kept in sync with roi_records for the running spacing check

for domain in DOMAINS:
    cand_idx = np.flatnonzero(domain_all == domain)
    order = rng.permutation(len(cand_idx))
    n_accepted, n_tried = 0, 0
    for oi in order:
        if n_accepted >= TARGET_ROIS_PER_DOMAIN:
            break
        n_tried += 1
        c = spatial_um[cand_idx[oi]]
        if accepted_centers_um:
            ac = np.asarray(accepted_centers_um)
            if np.min(np.hypot(ac[:, 0] - c[0], ac[:, 1] - c[1])) < MIN_CENTER_DIST_UM:
                continue
        nbr = np.asarray(tree_all.query_ball_point(c, r=ROI_RADIUS_UM))
        if len(nbr) < 5:
            continue
        purity = float(np.mean(domain_all[nbr] == domain))
        if purity < PURITY_THRESHOLD:
            continue
        roi_records.append(
            {
                "roi_id": len(roi_records) + 1,
                "domain": domain,
                "center_x_um": float(c[0]),
                "center_y_um": float(c[1]),
                "purity": purity,
            }
        )
        accepted_centers_um.append(c)
        n_accepted += 1
    checkpoint(
        f"  {domain:16s} candidates={len(cand_idx):7d} tried={n_tried:6d} "
        f"accepted={n_accepted:3d}/{TARGET_ROIS_PER_DOMAIN}"
    )

rois = pd.DataFrame(roi_records)
checkpoint(f"Placed {len(rois)} ROIs total across {len(DOMAINS)} domains")

# ---------------------------------------------------------------------------
# 4. Assign cells to ROIs. Spacing already guarantees each cell falls in at most one ROI's
#    radius-R disk (min center distance >= 2R), so no de-duplication is needed.
# ---------------------------------------------------------------------------
valid_label = (
    ~np.isin(domain_all, list(EXCLUDE_LABELS)) & ~np.isin(cell_type_all, list(EXCLUDE_LABELS))
)

cell_rows = []
for _, roi in rois.iterrows():
    center = (roi["center_x_um"], roi["center_y_um"])
    nbr = np.asarray(tree_all.query_ball_point(center, r=ROI_RADIUS_UM))
    nbr = nbr[valid_label[nbr]]
    for i in nbr:
        cell_rows.append(
            {
                "cell_id": cell_id[i],
                "cell_type": cell_type_all[i],
                "cell_type_minor": cell_type_minor_all[i] if cell_type_minor_all is not None else "",
                "domain": domain_all[i],
                "roi_id": int(roi["roi_id"]),
                "roi_domain": roi["domain"],
                "roi_purity": roi["purity"],
                "x_he": he_px[i, 0],
                "y_he": he_px[i, 1],
                "_row": i,
            }
        )
cells = pd.DataFrame(cell_rows)
n_full = len(cells)
cells = cells.sample(frac=SUBSAMPLE_FRAC, random_state=RNG_SEED).sort_index().reset_index(drop=True)
checkpoint(f"TEST RUN: subsampled {n_full} -> {len(cells)} cells ({SUBSAMPLE_FRAC:.0%})")
n_cells_by_domain = cells.groupby("roi_domain")["cell_id"].count()
rois["n_cells"] = rois["roi_id"].map(cells.groupby("roi_id")["cell_id"].count()).fillna(0).astype(int)
checkpoint(
    f"Selected {len(cells)} cells across {len(rois)} ROIs "
    f"(excluded cells with domain/cell_type in {sorted(EXCLUDE_LABELS)})\n"
    + n_cells_by_domain.to_string()
)

# ---------------------------------------------------------------------------
# 5. One shared, cell-centred tile set (restricted to ROI-selected cells)
# ---------------------------------------------------------------------------
tile_spec = TileSpec.from_wsidata(wsi, tile_px=TILE_PX, mpp=TILE_MPP)
checkpoint(f"Tile spec: {tile_spec}")
half_w, half_h = tile_spec.base_width / 2, tile_spec.base_height / 2
xys = np.c_[cells["x_he"].to_numpy() - half_w, cells["y_he"].to_numpy() - half_h]

tile_key = "roi_cell_tiles"
add_tiles(wsi, key=tile_key, xys=xys, tile_spec=tile_spec, tissue_ids=np.zeros(len(xys), dtype=int))
wsi.shapes[tile_key]["cell_id"] = cells["cell_id"].to_numpy()
cells["tile_id"] = wsi.shapes[tile_key]["tile_id"].to_numpy()
checkpoint(f"Registered {len(xys)} cell-centred tiles")

# ---------------------------------------------------------------------------
# 6. Feature extraction with the foundation models (no downstream classifier -- see docstring)
# ---------------------------------------------------------------------------
embeddings = {}
for model_key in MODEL_KEYS:
    checkpoint(f"Extracting features: {model_key} ...")
    zs.tl.feature_extraction(wsi, model_key, tile_key=tile_key, amp=True)
    feat_adata = wsi.fetch.features_anndata(model_key, tile_key=tile_key, tile_graph=False)
    order = np.argsort(feat_adata.obs["tile_id"].to_numpy())
    assert np.array_equal(
        feat_adata.obs["tile_id"].to_numpy()[order], cells["tile_id"].to_numpy()
    ), f"tile order mismatch for {model_key}"
    embeddings[model_key] = np.asarray(feat_adata.X)[order]
    checkpoint(f"Done: {model_key}  embeddings shape={embeddings[model_key].shape}")

# ---------------------------------------------------------------------------
# 7. Save per-cell embeddings h5ad. No train/test split here -- split BY roi_id in Jupyter.
# ---------------------------------------------------------------------------
cell_adata = ad.AnnData(
    X=np.zeros((len(cells), 1), dtype=np.float32),  # placeholder; embeddings live in obsm
    obs=cells.drop(columns=["_row"]).set_index("cell_id", drop=False),
)
for model_key in MODEL_KEYS:
    cell_adata.obsm[model_key] = embeddings[model_key]
h5ad_path = out_dir / "ovca_roi_cell_embeddings.h5ad"
cell_adata.write_h5ad(h5ad_path)
checkpoint(f"Saved cell embeddings to {h5ad_path}")

# ---------------------------------------------------------------------------
# 8. Save ROI definitions as real circle polygons, in the SAME "global" coordinate system as
#    sdata.images["he"] / sdata.labels["cell_labels"], so they can be attached straight back:
#        from spatialdata.models import ShapesModel
#        from spatialdata.transformations import Identity
#        gdf = gpd.read_file(".../ovca_roi_definitions.geojson")
#        sdata.shapes["roi_definitions"] = ShapesModel.parse(gdf, transformations={"global": Identity()})
#        sdata.pl.render_images("he").pl.render_shapes("roi_definitions").pl.show()
#    (a plain GeoDataFrame can't be assigned to sdata.shapes directly -- spatialdata requires
#    the ShapesModel.parse(..., transformations=...) wrapping above; verified against this repo's
#    spatialdata version.) he-px columns are included too, for a plain-matplotlib overlay on the
#    raw H&E array without going through spatialdata at all.
# ---------------------------------------------------------------------------
roi_centers_um = rois[["center_x_um", "center_y_um"]].to_numpy()
roi_centers_global_px = roi_centers_um / PIXEL_SIZE
roi_centers_he_px = (
    np.c_[roi_centers_global_px, np.ones(len(roi_centers_global_px))] @ np.linalg.inv(he_to_global).T
)[:, :2]
radius_global_px = ROI_RADIUS_UM / PIXEL_SIZE
radius_he_px = ROI_RADIUS_UM / HE_MPP

roi_gdf = gpd.GeoDataFrame(
    {
        "roi_id": rois["roi_id"].to_numpy(),
        "roi_domain": rois["domain"].to_numpy(),
        "purity": rois["purity"].to_numpy(),
        "n_cells": rois["n_cells"].to_numpy(),
        "radius_um": ROI_RADIUS_UM,
        "radius_global_px": radius_global_px,
        "center_x_he": roi_centers_he_px[:, 0],
        "center_y_he": roi_centers_he_px[:, 1],
        "radius_he_px": radius_he_px,
    },
    geometry=[Point(xy).buffer(radius_global_px) for xy in roi_centers_global_px],
)
geojson_path = out_dir / "ovca_roi_definitions.geojson"
roi_gdf.to_file(geojson_path, driver="GeoJSON")
checkpoint(f"Saved {len(roi_gdf)} ROI polygons to {geojson_path}")

# ---------------------------------------------------------------------------
# 9. QC plot: all cells (faint, by domain) + accepted ROI circles overlaid, in H&E px
# ---------------------------------------------------------------------------
domain_palette = dict(zip(DOMAINS, plt.cm.tab10.colors))
domain_palette["other"] = (0.85, 0.85, 0.85)

plot_idx = np.arange(len(he_px))
if len(plot_idx) > QC_PLOT_MAX_CELLS:
    plot_idx = np.sort(rng.choice(len(he_px), size=QC_PLOT_MAX_CELLS, replace=False))
plot_domain = np.where(np.isin(domain_all[plot_idx], DOMAINS), domain_all[plot_idx], "other")

fig, ax = plt.subplots(figsize=(12, 16))
for d in ["other"] + DOMAINS:
    m = plot_domain == d
    ax.scatter(
        he_px[plot_idx][m, 0], he_px[plot_idx][m, 1],
        s=1, alpha=0.15 if d == "other" else 0.35, color=domain_palette[d], label=d,
    )
for _, roi in roi_gdf.iterrows():
    ax.add_patch(
        Circle(
            (roi["center_x_he"], roi["center_y_he"]), roi["radius_he_px"],
            fill=False, edgecolor=domain_palette.get(roi["roi_domain"], "black"), linewidth=1.5,
        )
    )
ax.invert_yaxis()
ax.set_aspect("equal")
ax.set_title(f"OVCA ROI placement: {len(roi_gdf)} ROIs, R={ROI_RADIUS_UM}um, purity>={PURITY_THRESHOLD}")
ax.legend(markerscale=8, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout()
qc_path = out_dir / "ovca_roi_qc_overlay.png"
fig.savefig(qc_path, dpi=150, bbox_inches="tight")
plt.close(fig)
checkpoint(f"Saved QC overlay plot to {qc_path}")

# ---------------------------------------------------------------------------
# 10. Manifest: config + actual vs target coverage, for a glance without opening the h5ad
# ---------------------------------------------------------------------------
manifest = {
    "config": {
        "model_keys": MODEL_KEYS,
        "tile_px": TILE_PX,
        "tile_mpp": TILE_MPP,
        "tile_side_um": TILE_SIDE_UM,
        "he_mpp": HE_MPP,
        "pixel_size_um_per_global_px": PIXEL_SIZE,
        "roi_radius_um": ROI_RADIUS_UM,
        "min_roi_center_spacing_um": MIN_CENTER_DIST_UM,
        "purity_threshold": PURITY_THRESHOLD,
        "target_rois_per_domain": TARGET_ROIS_PER_DOMAIN,
        "domains": DOMAINS,
        "rng_seed": RNG_SEED,
    },
    "coverage": {
        d: {
            "n_rois": int((rois["domain"] == d).sum()),
            "target_rois": TARGET_ROIS_PER_DOMAIN,
            "n_cells": int(n_cells_by_domain.get(d, 0)),
            "mean_purity": float(rois.loc[rois["domain"] == d, "purity"].mean()) if (rois["domain"] == d).any() else None,
        }
        for d in DOMAINS
    },
    "total_rois": int(len(rois)),
    "total_cells": int(len(cells)),
    "outputs": {
        "cell_embeddings_h5ad": str(h5ad_path),
        "roi_definitions_geojson": str(geojson_path),
        "qc_overlay_png": str(qc_path),
    },
}
manifest_path = out_dir / "ovca_roi_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=float)
checkpoint("Coverage summary:\n" + json.dumps(manifest["coverage"], indent=2))
checkpoint(f"Saved manifest to {manifest_path}")
checkpoint(f"All artifacts saved under {out_dir}")
checkpoint("Done.")
