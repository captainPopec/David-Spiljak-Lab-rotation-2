"""Make a XOA-3.0 zarr-packed Xenium bundle readable by spatialdata_io.

Newer 10x Xenium Onboard Analysis bundles (the ``*_xe_outs.zip`` on the dataset page,
version 3.0.0+) ship the outputs ONLY as nested ``*.zarr.zip`` archives:

    experiment.xenium
    cell_feature_matrix.zarr.zip     <- expression (CSC), NOT cell_feature_matrix.h5
    cells.zarr.zip                   <- masks, polygons, cell_summary, cell_id
    transcripts.zarr.zip
    morphology_focus/*.ome.tif

``spatialdata_io.xenium`` (all versions through main, 2026) still reads the *legacy flat
files* — ``cell_feature_matrix.h5`` and ``cells.parquet`` — which this bundle no longer
contains. It DOES read masks/polygons/cell_summary from ``cells.zarr.zip`` natively.

This module materialises the two missing flat files from the zarr, so the stock reader runs
unmodified:
  * ``cell_feature_matrix.h5``  from ``cell_feature_matrix.zarr.zip`` (its ``csc/`` subgroup is
    already in 10x CSC orientation: features x cells, indptr over cells).
  * ``cells.parquet``           from ``cells.zarr.zip`` ``cell_summary`` + decoded ``cell_id``.

Feature types come from ``gene_panel.json``: the 5,101 panel targets are ``Gene Expression``;
the remaining features (neg-control probes/codewords, deprecated codewords, ...) are marked as
controls so ``sc.read_10x_h5(gex_only=True)`` drops them. Barcodes are decoded with
spatialdata_io's own ``cell_id_str_from_prefix_suffix_uint32`` so they match, exactly, the
``cell_id`` the reader derives from ``cells.zarr.zip`` (which the reader cross-checks).

Idempotent: does nothing if the flat files already exist. Boundary parquets are NOT
synthesised (the notebook renders the ``cell_labels`` raster, which the reader builds from the
zarr); ``build_ovca.py`` skips boundaries when their parquets are absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import zarr

from spatialdata_io.readers.xenium import cell_id_str_from_prefix_suffix_uint32


def _zopen(zip_path: Path):
    return zarr.open(zarr.storage.ZipStore(str(zip_path), read_only=True), mode="r")


def _feature_types(feature_keys: list[str], gene_panel_path: Path) -> np.ndarray:
    """Gene Expression iff the feature is a panel target; else a control label."""
    panel = json.load(open(gene_panel_path))
    genes: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            t = o.get("type")
            if isinstance(t, dict) and t.get("descriptor") == "gene":
                nm = t.get("data", {}).get("name")
                if nm:
                    genes.add(nm)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(panel)
    if not genes:
        raise RuntimeError(f"No gene targets parsed from {gene_panel_path}; refusing to guess types.")

    def one(f: str) -> str:
        if f in genes:
            return "Gene Expression"
        if f.startswith("NegControlProbe") or f.startswith("antisense"):
            return "Negative Control Probe"
        if f.startswith("NegControlCodeword"):
            return "Negative Control Codeword"
        if f.startswith("UnassignedCodeword"):
            return "Unassigned Codeword"
        if "deprecated" in f.lower():
            return "Deprecated Codeword"
        if f.startswith("Intergenic") or "genomic" in f.lower():
            return "Genomic Control"
        return "Control"  # aggregate_gene, Total transcripts, ... -> anything != "Gene Expression"

    return np.array([one(f) for f in feature_keys], dtype=object)


def write_cell_feature_matrix_h5(bundle: Path) -> Path:
    """Write CellRanger-v3-format cell_feature_matrix.h5 from cell_feature_matrix.zarr.zip."""
    out = bundle / "cell_feature_matrix.h5"
    if out.exists():
        return out
    g = _zopen(bundle / "cell_feature_matrix.zarr.zip")["cell_features"]
    feature_keys = list(g.attrs["feature_keys"])
    n_features = len(feature_keys)

    csc = g["csc"]  # features x cells, CSC (indptr over cells) == 10x h5 layout
    data = np.asarray(csc["data"][...], dtype=np.int32)
    indices = np.asarray(csc["indices"][...], dtype=np.int64)   # feature row indices
    indptr = np.asarray(csc["indptr"][...], dtype=np.int64)     # len n_cells+1
    n_cells = len(indptr) - 1

    cid = g["cell_id"][...]
    barcodes = np.asarray(cell_id_str_from_prefix_suffix_uint32(cid[:, 0], cid[:, 1]), dtype=object)
    assert len(barcodes) == n_cells, f"barcodes {len(barcodes)} != n_cells {n_cells}"

    ftypes = _feature_types(feature_keys, bundle / "gene_panel.json")
    n_gex = int((ftypes == "Gene Expression").sum())
    print(f"[h5] {n_features} features ({n_gex} Gene Expression), {n_cells} cells, "
          f"{len(data):,} nonzeros", flush=True)

    S = lambda arr: np.array([str(x).encode() for x in arr], dtype="S")  # noqa: E731
    with h5py.File(out, "w") as f:
        m = f.create_group("matrix")
        m.create_dataset("data", data=data, compression="gzip")
        m.create_dataset("indices", data=indices, compression="gzip")
        m.create_dataset("indptr", data=indptr, compression="gzip")
        m.create_dataset("shape", data=np.array([n_features, n_cells], dtype=np.int64))
        m.create_dataset("barcodes", data=S(barcodes), compression="gzip")
        ft = m.create_group("features")
        ft.create_dataset("id", data=S(feature_keys), compression="gzip")
        ft.create_dataset("name", data=S(feature_keys), compression="gzip")
        ft.create_dataset("feature_type", data=S(ftypes), compression="gzip")
        ft.create_dataset("genome", data=S([""] * n_features), compression="gzip")
        ft.create_dataset("_all_tag_keys", data=np.array([b"genome"], dtype="S"))
    print(f"[h5] wrote {out}", flush=True)
    return out


def write_cells_parquet(bundle: Path) -> Path:
    """Write cells.parquet (cell_id + centroids + areas) from cells.zarr.zip cell_summary."""
    out = bundle / "cells.parquet"
    if out.exists():
        return out
    cz = _zopen(bundle / "cells.zarr.zip")
    summary = cz["cell_summary"]
    cols = list(summary.attrs["column_names"])
    x = np.asarray(summary[...])
    df = pd.DataFrame(x, columns=cols)

    cid = cz["cell_id"][...]
    df.insert(0, "cell_id", np.asarray(cell_id_str_from_prefix_suffix_uint32(cid[:, 0], cid[:, 1])))

    # rename cell_summary columns to the flat-file schema the reader expects
    ren = {"cell_centroid_x": "x_centroid", "cell_centroid_y": "y_centroid"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for need in ("x_centroid", "y_centroid", "cell_area"):
        if need not in df.columns:
            raise RuntimeError(f"cell_summary missing {need!r}; columns were {cols}")
    df.to_parquet(out, index=False)
    print(f"[parquet] wrote {out}  ({len(df):,} cells, cols={list(df.columns)})", flush=True)
    return out


def ensure_flat_files(bundle: Path) -> None:
    """Materialise cell_feature_matrix.h5 + cells.parquet in `bundle` if they are missing."""
    bundle = Path(bundle)
    if (bundle / "cell_feature_matrix.zarr.zip").exists():
        write_cell_feature_matrix_h5(bundle)
    if (bundle / "cells.zarr.zip").exists():
        write_cells_parquet(bundle)


if __name__ == "__main__":
    import sys
    ensure_flat_files(Path(sys.argv[1]))
    print("DONE_COMPAT", flush=True)
