#!/bin/bash
#SBATCH --job-name=OVCA_roi_cell_emb
#SBATCH --partition=gpu-el8
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu=3090|gpu=A100|gpu=A40|gpu=L40s|gpu=H100"
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/g/stegle/spiljak/projects/main_wsi_ood/scripts/logs/extract_cell_embeddings_roi_%j.out
#SBATCH --error=/g/stegle/spiljak/projects/main_wsi_ood/scripts/logs/extract_cell_embeddings_roi_%j.err

# ROI-based (non-overlapping, per-domain) cell-tile embedding extraction -- see
# extract_cell_embeddings_roi.py docstring for the leakage problem this avoids.
#
# Sizing: with R=150um, purity>=0.60, target 12 ROIs/domain, ROI placement gives ~36 ROIs /
# ~17.3k cells total (vs. the old per-cell script's ~405k cells over the whole slide).
# The first attempt (job 59237872) timed out at 4h without even finishing virchow2 alone:
# root cause was that this build's OVCA.zarr `he` image was written as ONE giant zarr chunk
# (no tiling, no multiscale pyramid), so every scattered per-cell tile read was ~14x slower
# than the old goofys_wsi_ood build (which chunks `he_image` at 4096x4096 across a 5-level
# pyramid). Fixed by rechunking `he` in place (see rechunk_he.py in scratch, applied once).
# MODEL_KEYS is virchow2-only for this run (UNI2 dropped to save time -- run separately
# later); 36h time budget is a generous safety margin post-fix, not a real expected duration.
#
# virchow2 weights are already cached in this env (gated HF repo -- authenticate once with
# `huggingface-cli login` under this env if the cache is ever cleared).

mkdir -p /g/stegle/spiljak/projects/main_wsi_ood/scripts/logs

module load CUDA/12.1.0
source /g/stegle/spiljak/programs/miniforge3/etc/profile.d/conda.sh
conda activate lazyslide-env

python /g/stegle/spiljak/projects/main_wsi_ood/scripts/extract_cell_embeddings_roi.py
