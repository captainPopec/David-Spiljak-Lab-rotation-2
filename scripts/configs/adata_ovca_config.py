# OVCA AnnData preprocessing defaults (mirrors adata_crc_config.py for the OVCA LOO run).
#
# Differences from CRC, and why:
#   - n_top_genes=None: the Xenium panel is already a curated 5101-gene panel, not an
#     unbiased whole-transcriptome assay. seurat_v3 HVG selection exists to cut noise from
#     the latter; applying it here would just discard curated genes. So: no HVG subsetting.
#   - bandwidth_um: OVCA's obsm["spatial"] is in MICRONS, whereas CRC's coords are in pixels
#     (hence CRC's 100/step_size_px conversion). cellina's spatial_neighbors takes bandwidth
#     in the units of obsm["spatial"], so 100um is passed through directly.
#   - batch_key=None: a single slide, so there is no batch to correct (CRC used "sid").

ADATA_ARGS = {
    # no HVG subsetting -- keep the full curated Xenium panel
    "n_top_genes": None,
    # spatial neighbour graph
    "n_neighbors": 200,
    "bandwidth_um": 100.0,
    # minimum counts filtering
    "min_counts_cells": 3,
    "min_counts_genes": 3,
    "labels_key": "cell_type",
    # ground-truth domain: drives model setup + the train/val/test split
    "domains_key": "domain",
    # WSI-derived domain: drives the counterfactual arms + scoring
    "pred_domains_key": "pred_domain",
    "batch_key": None,
    "control_domains": ["Smooth muscle"],
    "holdout_domains": ["Tumor"],
    "holdout_celltype": "Stromal",
    # Tumor ROIs withheld ENTIRELY from training, for models conditioned on per-cell H&E tiles
    # (see eval.split_indices_roi_holdout). Tiles are ~112um wide and cells sit 7-9um apart, so
    # a cell-level split leaks >90%-overlapping pixels from train into test; only an ROI-level
    # split is pixel-disjoint (extract_cell_embeddings_roi.py placed the ROIs to guarantee it).
    #
    # These are the 4 Tumor ROIs with the most Stromal-in-Tumor cells (181/134/129/116 = 560
    # scored OOD cells, vs 13 for the worst pair). That is a STATISTICAL-POWER choice, selected
    # on cell abundance and never on any model's score, so it cannot bias one model against
    # another -- but it does narrow the OOD claim to stroma-rich tumour regions. Say so.
    # Unused by OVCA_03_loo.ipynb, whose models see no pixels.
    "holdout_rois": [9, 6, 3, 8],
}
