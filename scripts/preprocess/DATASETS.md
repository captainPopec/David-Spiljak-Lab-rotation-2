# SpatialFusion datasets — acquisition feasibility & coverage

Target use: extend `src/wsi/model.py` (WSICellina) with H&E/WSI image features via
**LazySlide**, which consumes **SpatialData**. So each dataset must end up as a
SpatialData object bundling image + cells (coords + cell-type + domain) + (ideally)
segmentation masks, saved under `notebooks/datasets/`.

Source paper: *SpatialFusion: A lightweight multimodal foundation model for
pathway-informed spatial niche mapping* (Yates et al., bioRxiv 2026,
DOI 10.64898/2026.03.16.712056). Method repo: `uhlerlab/spatialfusion`;
analysis repo: `uhlerlab/spatialfusion-analysis`.

## Requirements rubric

| ID | Requirement |
|----|-------------|
| **R1** | Paired H&E / WSI image, aligned to transcriptomics |
| **R2** | Cell-type annotations |
| **R3** | Spatial domain annotations — *ideally pathologist* |
| **R4** | Cell/nucleus segmentation masks |
| **R5** | Deliverable as a SpatialData object |

Legend: ✅ available · ⚠️ obtainable with extra work · ❌ not available

## Coverage table

| Dataset | Platform | Genes | Cells/bins | R1 image | R2 cell type | R3 domain (pathologist) | R4 segmentation | R5 SpatialData | Verdict |
|---------|----------|------:|------:|:--:|:--:|:--:|:--:|:--:|---------|
| **PRAD** prostate | Xenium Prime | 5,006¹ | 193,000 | ✅ H&E ome.tif | ✅ `celltypes_prad.csv` | ✅✅ **de novo board-certified pathologist** (QuPath geojson) | ✅ native Xenium boundaries | ✅ `sdio.xenium()` | **Best — meets all 5, gold R3** |
| **OVCA** ovarian | Xenium Prime | 5,101 | 407,124 | ✅ H&E ome.tif | ✅ `celltypes_ovca.csv` | ✅ regions (Tumor/Necrosis) — *retraced in QuPath from 10x-annotated image, not de novo* | ✅ native | ✅ | **Meets all 5; R3 secondhand** |
| **BRCA** breast | Xenium | 313 | 167,780 | ✅ H&E | ✅ `celltypes_brca.csv` (major+minor) | ✅ DCIS/invasive — *retraced from the Nat. Commun. paper figure* | ✅ native | ✅ | **Meets all 5; R3 secondhand; small panel** |
| **LUAD** lung | Visium HD | 18,028 | 204k bins | ✅ H&E .btf | ✅ (author) | ✅ grade 2/3/normal — *reconstructed from Long et al.* | ⚠️ **bin2cell rerun** (no native cells) | ✅ `sdio.visium_hd()` | Visium HD — dropped (R4) |
| **CRC** colorectal | Visium HD | ~18,000 | 5 samples | ✅ H&E .btf | ✅ (author) | ❌ ~none (dysplasia in 1 sample only) | ⚠️ bin2cell rerun | ✅ | **Fails R3** |
| **NSCLC** lung | Visium HD | ~18,000 | 10 in-house | ❌ | ❌ | ❌ (sample-level pTNM only) | ❌ | ❌ | **Embargoed — not public** |
| **HEST-1k Xenium** | Xenium | ~100–5,000 (panel) | ~11M / 51–65 | ✅ all | ⚠️ derivable (cluster), not shipped | ❌ **none ship pathologist regions** | ✅ native + CellViT nuclei | ✅ `to_spatial_data()` | **Fails R3 (its whole point is gene-prediction, not regions)** |

**Selected set (all-Xenium, all 5 requirements met): PRAD, OVCA, BRCA.**

¹ PRAD as-built has **5,006** gene-expression features from the public 10x bundle (the
Xenium Prime 5K panel), not the ~9,984 implied by the SpatialFusion deposit — flag if the
exact panel matters. Cell/bin counts above are **as-built** from the 10x bundles (the
deposit's smaller counts were the *annotated* subsets, not total cells).

## As-built results (SpatialData `.zarr` under `notebooks/datasets/`)

Built by `scripts/preprocess/build_{ovca,prad,brca}.py` (raw 10x bundle → `sdio.xenium()`
→ join author cell types + pathologist domains → write zarr). Raw staging deleted after
each build.

| Dataset | zarr | cell_type matched | domain matched | R3 element | H&E |
|---------|-----:|------------------:|---------------:|------------|-----|
| OVCA | 15 GB | 406,611/407,124 (99.9%) | 338,695/407,124 (83.2%) | `domain_points` + `obs['domain']` | `he`, affine-aligned |
| PRAD | 9 GB | 186,208/193,000 (96.5%) | 165,023/193,000 (85.5%) | `pathology_regions` shapes (gold geojson) + `obs['domain']` | `he_image`, affine-aligned |
| BRCA | 4 GB | 132,508/167,780 (79.0%) | 132,508/167,780 (79.0%) | `pathology_regions` shapes (dissolved per domain) + `obs['domain']` | `he`, affine-aligned |

Common elements per zarr: `images` (`he` + `morphology_focus`), `labels`
(`cell_labels`, `nucleus_labels`), `shapes` (`cell_boundaries`, `nucleus_boundaries`,
`pathology_regions`), `tables['table']` with `obs['cell_type']`, `cell_subtype`, `domain`.
Transcripts excluded to save space. Explore with `notebooks/04_explore_datasets.ipynb`.

## Per-dataset detail

### PRAD — prostate (Xenium Prime) — RECOMMENDED FLAGSHIP
- The only dataset with a **de novo** board-certified pathologist domain annotation.
- HEST id `TENX157` (also a direct 10x public dataset: *Xenium Prime Human Prostate FFPE*).
- Annotations in repo: `Fig4/celltypes_prad.csv` (cols `celltypes`,`cellsubtypes`, barcode `TENX157_…`), `Fig4/Xenium_Prime_Human_Prostate_FFPE_he_image.geojson` (QuPath polygons).
- Build: raw 10x Xenium bundle → `sdio.xenium()` (image+labels+shapes+table) → join cell types by barcode → rasterize/attach geojson regions.

### OVCA — ovarian (Xenium Prime)
- 10x public *Xenium Prime Ovarian Cancer FFPE*. Annotations: `Fig2/celltypes_ovca.csv`, `Fig2/region_annotations_ovca.csv` (`cell_id`,`Region_annotation`). Regions retraced from the 10x-provided annotated image.

### BRCA — breast (Xenium)
- HEST id `NCBI875` / 10x breast FFPE preview. Annotations: `Fig3/celltypes_brca.csv` (major+minor), `Fig3/breast_ffpe_region_annotations.csv` (DCIS2/Invasive/Unassigned). Smallest gene panel (~300).

### LUAD — lung (Visium HD)
- 10x *Visium HD Human Lung Cancer*. Region geojson in repo (`Fig4/…btf - 20x_BF_01.geojson`). **No native single-cell segmentation** — authors used bin2cell/StarDist; the Zenodo h5ad is **bin-level (204k bins)**, not cells. R4 requires rerunning bin2cell or accepting bins.

### CRC — colorectal (Visium HD) — fails R3
- 5 samples (P1CRC,P2CRC,P5CRC,P3NAT,P5NAT). Author cell types; essentially no pathologist regions. Segmentation via bin2cell. Include only if more Visium HD volume is wanted.

### NSCLC — in-house (Visium HD) — unavailable
- 10 samples (5 LUAD + 5 LUSC). "Associated data … TBD upon acceptance." Not reproducible now.

### HEST-1k Xenium subset — fails R3
- R1/R4/R5 ✅ (native Xenium boundaries + CellViT nuclei; `HESTData.to_spatial_data()`); R2 derivable; **R3 absent** — HEST's benchmark is gene-expression-from-histology, not region annotation. Useful only if you later want a large pretraining pool and will self-annotate domains in QuPath.

## Where the bytes live

- **Author annotations + cell types + PROGENy + preprocessed h5ad:** Zenodo. NOTE: two DOIs seen — `10.5281/zenodo.20934899` ("preprocessed benchmark dataset", 1.6 GB, the 4 benchmark `.h5ad`) and `10.5281/zenodo.18377525` ("cell-type annotations + PROGENy"). **Confirm which holds what.**
- **Raw 10x bundles (images + native segmentation):** 10x Genomics public datasets website, by name (Xenium Prime Ovarian/Prostate; Xenium breast FFPE; Visium HD Lung/Colon).
- **HEST-1k:** HuggingFace `MahmoodLab/hest` (gated; needs token), via `snapshot_download` + `hest.iter_hest`.
- **Region geojson + cell-type CSVs:** already committed in `uhlerlab/spatialfusion-analysis` (`Fig2…Fig6/`).

## Key technical notes

- **The Zenodo `.h5ad` alone is NOT enough** for this project: it has counts + coords +
  annotations but **no image and no segmentation**. Images/masks force the raw 10x bundles.
- **Segmentation (R4) splits by platform:** Xenium = native boundaries (free via
  `sdio.xenium()`); Visium HD = no native cells (need bin2cell, adds `bin2cell` to env).
- **Cell-ID join:** author barcodes are prefixed (`TENX157_aaaa…-1`); raw-bundle barcodes
  are bare (`aaaa…-1`). Joining annotations onto the SpatialData table needs a prefix
  strip. The Zenodo h5ad is the cleanest carrier of the joined annotations + coords.
- **Env:** the `wsi` conda env is built (spatialdata 0.7.3, spatialdata-io 0.7.0,
  geopandas 1.1.4, scanpy, lazyslide) + extras (tifffile, imagecodecs, zarr, dask).
  **libstdc++ caveat:** pip `scipy`/`lazyslide` need `CXXABI_1.3.15`, absent from the
  system `libstdc++`. The env's own newer `libstdc++` must be on the loader path. Fixed
  two ways: (a) `conda env config vars set` so `conda activate wsi` (and the Jupyter
  kernel) auto-sets `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`; (b) for direct calls, prefix:
  `LD_LIBRARY_PATH=/data/ddimitrov/software/miniforge3/envs/wsi/lib .../envs/wsi/bin/python …`.
  (`bin2cell` not installed — only needed for the dropped Visium HD datasets.)
- **Disk:** 305 GB free. Raw Xenium Prime bundles are large; budget per-dataset.

## Open decisions (being grilled)

1. Final dataset set (recommend: PRAD + OVCA + BRCA + LUAD).
2. R4 for Visium HD — rerun bin2cell vs accept bin-level vs skip masks.
3. Construction route — `spatialdata-io` from raw + join annotations (full, heavy) vs
   build-from-Zenodo-h5ad + attach image only (light, no native masks).
4. Which H&E image (post-Xenium aligned H&E vs morphology/DAPI).
5. Disk/download location + per-dataset budget.
6. Agent orchestration for download+convert.
