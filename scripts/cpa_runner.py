"""CPA (Compositional Perturbation Autoencoder) train + eval, as a competitor baseline.

Refactored out of the multi-model ``train_loo.py`` / ``eval_loo.py`` in
cellina-reproducibility into standalone functions so the notebook (and any script) can
``from cpa import train_cpa, cpa_reconstruct, cpa_counterfactual`` and run CPA on the
same LOO setup as cellina.

CPA is the perturbation model from Lotfollahi et al. (2023); the ``cpa`` package
(``cpa-tools``) lives in its own conda env — see ``envs/cpa_env.yml``.

Conventions (CRC task): the spatial domain (``typ_clean``: REF / TVA / CRC) is treated
as the CPA "perturbation", the control group is ``REF``, and the cell type
(``coarse_type``) is a categorical covariate. A dummy ``dose`` of 1.0 is added for API
compatibility.
"""

from __future__ import annotations

import numpy as np

COUNTS_PER_K = 1e4


def _to_array(x):
    if x is None:
        return None
    toarray = getattr(x, "toarray", None)
    if callable(toarray):
        return toarray()
    return np.asarray(x)


def _normalize_counts(x, eps=1e-8, scale=COUNTS_PER_K):
    """CPM-style normalisation to a fixed library size."""
    x = _to_array(x)
    return x / (x.sum(axis=1, keepdims=True) + eps) * scale


def _annotate_splits(adata, splits):
    """Write CPA's expected ``dose`` / ``data_split`` columns from (train, val, test) idx."""
    adata.obs["dose"] = 1.0  # dummy dose for compatibility with CPA
    adata.obs["data_split"] = "train"
    _, val_idx, test_idx = splits
    adata.obs.iloc[val_idx, adata.obs.columns.get_loc("data_split")] = "valid"
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc("data_split")] = "test"


def setup_cpa(adata, splits, domains_key="typ_clean", labels_key="coarse_type", control_group="REF"):
    """Annotate splits and register the AnnData with CPA. Mutates ``adata`` in place."""
    import cpa

    _annotate_splits(adata, splits)
    cpa.CPA.setup_anndata(
        adata,
        perturbation_key=domains_key,
        control_group=control_group,
        dosage_key="dose",
        categorical_covariate_keys=[labels_key],
        is_count_data=True,
        max_comb_len=1,
    )


def train_cpa(adata, model_args, train_args, plan_kwargs, save_dir, splits,
              domains_key="typ_clean", labels_key="coarse_type", control_group="REF"):
    """Set up, train, and save a CPA model. Returns the trained model."""
    import cpa

    setup_cpa(adata, splits, domains_key=domains_key, labels_key=labels_key, control_group=control_group)
    model = cpa.CPA(
        adata,
        split_key="data_split",
        train_split="train",
        valid_split="valid",
        test_split="test",
        **model_args,
    )
    model.train(**train_args, plan_kwargs=plan_kwargs, save_path=save_dir)
    return model


def load_cpa(save_dir, adata, splits,
             domains_key="typ_clean", labels_key="coarse_type", control_group="REF"):
    """Re-register the AnnData and load a trained CPA model from ``save_dir``."""
    import cpa

    setup_cpa(adata, splits, domains_key=domains_key, labels_key=labels_key, control_group=control_group)
    return cpa.CPA.load(dir_path=save_dir, adata=adata, use_gpu=True)


def cpa_reconstruct(model, adata_obj, return_normalized=False, batch_size=512):
    """Reconstruct expression for ``adata_obj`` via ``model.predict`` -> ``obsm['CPA_pred']``."""
    out = model.predict(adata_obj, batch_size=batch_size)
    if "CPA_pred" in adata_obj.obsm:
        X = _to_array(adata_obj.obsm["CPA_pred"])  # count-like predictions
        return _normalize_counts(X) if return_normalized else X
    if out is not None:
        return _to_array(out)
    raise RuntimeError("CPA produced no output and did not populate adata.obsm['CPA_pred']")


def cpa_counterfactual(model, adata, idx_control, holdout_domain, return_normalized=False, batch_size=512):
    """Predict the counterfactual: control cells (``idx_control``) moved to ``holdout_domain``.

    Mirrors how CPA generates counterfactuals: subset the control cells, overwrite their
    perturbation code with the target domain's index, flag them as non-control, and decode.
    """
    from cpa._utils import CPA_REGISTRY_KEYS

    adata_ctrl = adata[idx_control].copy()
    adata_ctrl.obsm["perts"][:, 0] = model.pert_encoder[holdout_domain]
    adata_ctrl.obs[CPA_REGISTRY_KEYS.CONTROL_KEY] = 0
    return cpa_reconstruct(model, adata_ctrl, return_normalized=return_normalized, batch_size=batch_size)
