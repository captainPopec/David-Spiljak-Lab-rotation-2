"""Input-influence ablation for the OVCA LOO decoders: how much does each INPUT move the output?

The OVCA LOO models are decoders with no encoder. After library size is factored out
(``library_size=1e4`` -> px_scale * 1e4), a prediction depends on two inputs only:

    LabelsBaseline      f(cell type, domain one-hot)
    EmbeddingsBaseline  f(cell type, H&E embedding)       Virchow2 or UNI2

So the question "how much of the output is driven by the cell-type label vs by the embedding"
has an exact answer: decode a FULL FACTORIAL grid -- every cell type x every context (donor
embedding, or domain) -- and split the variance of the log-expression with a balanced two-way
ANOVA, per gene, summed over genes:

    total = cell type (main effect) + context (main effect) + interaction

For context, the shared profile (the gene-to-gene variation of the grand-mean profile) is also
reported. It is identical for every grid point, so it is not attributed to either input.

No retraining: the saved models in ``PRED_DIR/models`` are loaded and decoded on CPU.

Context panels
--------------
- ``donor_tumor_rois``: N_DONORS embeddings drawn from the SAME pool the counterfactual uses
  (non-Stromal Tumor cells in the held-out ROIs 3/6/8/9). The question is how much the random
  donor moves the prediction, compared with the cell type.
- ``donor_all_tissues``: N_PER_DOMAIN embeddings from each of the 5 domains (non-Stromal, any
  ROI). The question is how much the embedding moves the prediction across tissues.
- Labels: the 5 domain one-hots (the only contexts that model has).

The same donor cells are used for every model and seed. Cell types = every type the model was
trained on (7), weighted equally (balanced design).

Outputs (``PRED_DIR/influence/``)
---------------------------------
- ``anova.csv``: one row per (model, seed, panel): SS shares.
- ``grid_{model}_seed{s}_{panel}.npz``: the decoded grid, log1p(CPM 1e4), float16,
  shape (n_types, n_contexts, n_genes), plus types / context ids / genes, so any downstream
  figure can be rebuilt without decoding again.

Run with the ``cellina_edge`` env (needs scvi + the ``wsi`` package):
    srun -p htc-el8 --mem=48G -c 8 -t 01:00:00 \
        /g/stegle/spiljak/programs/miniforge3/envs/cellina_edge/bin/python \
        scripts/input_influence_ablation.py
"""

import os
import sys

import numpy as np
import pandas as pd
from pandas.core.arrays.string_ import StringDtype
from sklearn.preprocessing import StandardScaler

# The 2026-08-02 labels/Virchow2 models were saved under pandas >= 2.3, which pickles StringDtype
# as StringDtype(storage, na_value). cellina_edge has pandas 2.2.3, whose __init__ takes storage
# only, so torch.load dies in the model's saved registry (gene/obs names, not the weights).
# Accept and drop na_value; the names themselves are plain strings either way.
_string_dtype_init = StringDtype.__init__


def _string_dtype_init_compat(self, storage=None, na_value=None):
    _string_dtype_init(self, storage)


StringDtype.__init__ = _string_dtype_init_compat

PROJECT_ROOT = "/g/stegle/spiljak/projects/main_wsi_ood"
sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from configs.adata_ovca_config import ADATA_ARGS  # noqa: E402
from configs.embeddings_config import SEEDS  # noqa: E402
from eval import load_ovca_loo_bundle, preprocess_ovca, split_indices_roi_holdout  # noqa: E402
from wsi.embeddings_baseline import EmbeddingsBaseline  # noqa: E402
from wsi.labels_baseline import LabelsBaseline  # noqa: E402

# The saved models were written by a newer scvi-tools, which adds an (empty) "pyro_param_store"
# entry to every module's state dict. These decoders have no pyro parameters, so that one key is
# dropped and loading stays strict for everything else. The reproduction check below confirms
# that the loaded weights give back the cached predictions.
from wsi.embeddings_baseline import EmbeddingsBaselineModule  # noqa: E402
from wsi.labels_baseline import LabelsBaselineModule  # noqa: E402


def _drop_pyro_store(load_state_dict):
    def wrapped(self, state_dict, *args, **kwargs):
        state_dict = {k: v for k, v in state_dict.items() if k != "pyro_param_store"}
        return load_state_dict(self, state_dict, *args, **kwargs)
    return wrapped


for _m in (EmbeddingsBaselineModule, LabelsBaselineModule):
    _m.load_state_dict = _drop_pyro_store(_m.load_state_dict)

BUNDLE_DIR = os.path.join(PROJECT_ROOT, "data/ovca_loo")
PRED_DIR = "/scratch/spiljak/wsi_ood_preds/ovca_loo"
MODEL_DIR = os.path.join(PRED_DIR, "models")
OUT_DIR = os.path.join(PRED_DIR, "influence")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS_KEY = ADATA_ARGS["labels_key"]
DOMAINS_KEY = ADATA_ARGS["domains_key"]
HOLDOUT_CT = ADATA_ARGS["holdout_celltype"]
HOLDOUT_LABEL = ADATA_ARGS["holdout_domains"][0]
HOLDOUT_ROIS = ADATA_ARGS["holdout_rois"]

N_DONORS = 300          # donor_tumor_rois panel
N_PER_DOMAIN = 60       # donor_all_tissues panel (x 5 domains = 300)
DONOR_SEED = 0          # one fixed donor set for every model and seed
EMBEDDINGS = {"virchow2": "", "uni2": "uni2_"}   # obsm key -> model-dir prefix


# --- data, exactly as OVCA_03_loo_with_baseline.ipynb section A -------------------------------
adata = load_ovca_loo_bundle(BUNDLE_DIR)
adata = preprocess_ovca(adata, n_top_genes=ADATA_ARGS["n_top_genes"], labels_key=LABELS_KEY,
                        domains_key=DOMAINS_KEY, pred_domains_key=ADATA_ARGS["pred_domains_key"])
train_idx, _, _ = split_indices_roi_holdout(
    adata, HOLDOUT_CT, HOLDOUT_ROIS, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY,
    holdout_domains=ADATA_ARGS["holdout_domains"], seed=0)
for emb in EMBEDDINGS:
    scaler = StandardScaler().fit(np.asarray(adata.obsm[emb])[train_idx])
    adata.obsm[f"{emb}_scaled"] = np.ascontiguousarray(
        scaler.transform(np.asarray(adata.obsm[emb])).astype(np.float32))
print(adata.shape, "train cells for the scaler:", len(train_idx))

ct = adata.obs[LABELS_KEY].to_numpy(dtype=str)
dom = adata.obs[DOMAINS_KEY].to_numpy(dtype=str)
roi = adata.obs["roi_id"].to_numpy()
TYPES = list(adata.obs[LABELS_KEY].cat.categories)
DOMAINS = list(adata.obs[DOMAINS_KEY].cat.categories)
GENES = np.asarray(adata.var_names)

# --- donor panels (same cells for every model) ------------------------------------------------
rng = np.random.default_rng(DONOR_SEED)
not_holdout_ct = ct != HOLDOUT_CT
pool_tumor = np.where((dom == HOLDOUT_LABEL) & np.isin(roi, HOLDOUT_ROIS) & not_holdout_ct)[0]
PANELS = {
    "donor_tumor_rois": rng.choice(pool_tumor, size=N_DONORS, replace=False),
    "donor_all_tissues": np.concatenate([
        rng.choice(np.where((dom == d) & not_holdout_ct)[0], size=N_PER_DOMAIN, replace=False)
        for d in DOMAINS]),
}
for name, idx in PANELS.items():
    print(f"{name}: {len(idx)} donors; domains {pd.Series(dom[idx]).value_counts().to_dict()}")

# Any cell can carry the grid: with library_size=1e4 its counts never enter the output.
BASE = int(np.where(dom == ADATA_ARGS["control_domains"][0])[0][0])


def grid_adata(n_types, n_ctx):
    """n_types * n_ctx copies of the base cell, type-major (row = t * n_ctx + c)."""
    g = adata[np.full(n_types * n_ctx, BASE)].copy()
    g.obs_names = [f"grid{i}" for i in range(g.n_obs)]
    g.obs[LABELS_KEY] = pd.Categorical(np.repeat(TYPES, n_ctx), categories=TYPES)
    return g


def decode(model, g):
    x = model.get_normalized_expression(adata=g, library_size=1e4, batch_size=4096)
    return np.log1p(x).astype(np.float64)


def anova(Y):
    """Balanced two-way variance split of Y (types, contexts, genes), summed over genes."""
    m = Y.mean((0, 1))
    a = Y.mean(1) - m                     # cell-type main effect   (T, G)
    b = Y.mean(0) - m                     # context main effect     (C, G)
    r = Y - m - a[:, None] - b[None]      # interaction             (T, C, G)
    T, C, _ = Y.shape
    ss = {"type": C * (a ** 2).sum(), "context": T * (b ** 2).sum(), "interaction": (r ** 2).sum()}
    total = sum(ss.values())
    shared = T * C * ((m - m.mean()) ** 2).sum()
    return {**{f"frac_{k}": v / total for k, v in ss.items()},
            "ss_total": total, "frac_shared_of_all": shared / (shared + total)}


def save_grid(tag, seed, panel, Y, ctx_ids):
    np.savez_compressed(os.path.join(OUT_DIR, f"grid_{tag}_seed{seed}_{panel}.npz"),
                        Y=Y.astype(np.float16), types=np.array(TYPES),
                        context=np.asarray(ctx_ids).astype(str), genes=GENES)


import h5py  # noqa: E402

mask_control = dom == ADATA_ARGS["control_domains"][0]
idx_control = np.where(mask_control)[0]


def check_reproduces(model, cached_key, seed, cf):
    """Max |diff| in CPM between a fresh counterfactual and the cached one (perturbed arm)."""
    with h5py.File(os.path.join(PRED_DIR, f"{cached_key}_seed{seed}.h5ad"), "r") as f:
        cached = f["X"][-len(idx_control):]          # comparison AnnData = [control; perturbed]
    cpm = lambda x: 1e4 * x / x.sum(1, keepdims=True)
    d = np.abs(cpm(cf) - cpm(cached)).max()
    print(f"  reproduce {cached_key}_seed{seed}: max |dCPM| = {d:.2e}")
    assert d < 1e-2, f"{cached_key} seed {seed} does not reproduce its cached prediction"


rows = []
for seed in SEEDS:
    # Labels: cell type x the 5 domain one-hots
    LabelsBaseline.setup_anndata(adata, labels_key=LABELS_KEY, domains_key=DOMAINS_KEY,
                                 batch_key=ADATA_ARGS["batch_key"], layer="counts")
    model = LabelsBaseline.load(os.path.join(MODEL_DIR, f"labels_seed{seed}"), adata=adata,
                                accelerator="cpu")
    check_reproduces(model, "labels", seed, model.get_counterfactual_expression(
        idx_control, target_domain=HOLDOUT_LABEL, batch_size=4096))
    g = grid_adata(len(TYPES), len(DOMAINS))
    g.obs[DOMAINS_KEY] = pd.Categorical(np.tile(DOMAINS, len(TYPES)), categories=DOMAINS)
    Y = decode(model, g).reshape(len(TYPES), len(DOMAINS), -1)
    save_grid("labels", seed, "domains", Y, DOMAINS)
    rows.append({"model": "Labels", "seed": seed, "panel": "domains", "n_contexts": len(DOMAINS),
                 **anova(Y)})
    print(rows[-1])

    for emb, prefix in EMBEDDINGS.items():
        EmbeddingsBaseline.setup_anndata(adata, labels_key=LABELS_KEY,
                                         embedding_obsm_key=f"{emb}_scaled",
                                         batch_key=ADATA_ARGS["batch_key"], layer="counts")
        model = EmbeddingsBaseline.load(os.path.join(MODEL_DIR, f"{prefix}embeddings_seed{seed}"),
                                        adata=adata, accelerator="cpu")
        donors_cf = np.load(os.path.join(PRED_DIR, f"donors_{prefix}emb_unseen_ROIs_seed{seed}.npy"))
        check_reproduces(model, f"{prefix}emb_unseen_ROIs", seed, model.get_counterfactual_expression(
            idx_control, donor_indices=donors_cf, batch_size=4096))
        emb_all = np.asarray(adata.obsm[f"{emb}_scaled"])
        for panel, donors in PANELS.items():
            g = grid_adata(len(TYPES), len(donors))
            g.obsm[f"{emb}_scaled"] = np.ascontiguousarray(np.tile(emb_all[donors], (len(TYPES), 1)))
            Y = decode(model, g).reshape(len(TYPES), len(donors), -1)
            name = {"virchow2": "Virchow2", "uni2": "UNI2"}[emb]
            save_grid(name.lower(), seed, panel, Y, adata.obs_names[donors])
            rows.append({"model": name, "seed": seed, "panel": panel, "n_contexts": len(donors),
                         **anova(Y)})
            print(rows[-1])

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT_DIR, "anova.csv"), index=False)
print("\n", res.groupby(["model", "panel"])[
    ["frac_type", "frac_context", "frac_interaction", "frac_shared_of_all"]].agg(["mean", "std"]).round(3))
print("wrote", OUT_DIR)
