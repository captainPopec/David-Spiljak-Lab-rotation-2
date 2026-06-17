# Cellina default configuration.
# Ported from cellina-reproducibility/scripts/configs/cellina_config.py
#
# MODEL_ARGS are passed to ``cellina.Cellina(adata, **MODEL_ARGS)``.
# TRAIN_ARGS / PLAN_KWARGS are passed to ``model.train(**TRAIN_ARGS, plan_kwargs=PLAN_KWARGS)``
# (``datasplitter_kwargs['external_indexing']`` is filled in at runtime with the splits).

MODEL_ARGS = {
    "n_latent": 64,
    "use_observed_lib_size": True,
    "condition_on_intrinsic": False,
    "classifier_lambda": 1.0,      # lambda_clf  (cell-type classifier on z)
    "discriminator_lambda": 1.0,   # lambda_adv  (domain adversary on z)
    "gene_likelihood": "nb",
    "n_layers": 2,
}

TRAIN_ARGS = {
    "max_epochs": 100,
    "batch_size": 2048,
    "check_val_every_n_epoch": 1,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_monitor": "vae_loss_validation",
    "enable_checkpointing": True,
    "devices": [0],
}

PLAN_KWARGS = {
    "lr": 1e-3,
    "normalize_losses": True,  # compute per-loss normalization scales (alpha) over epoch 0
}

DO_COUNTERFACTUAL = True
