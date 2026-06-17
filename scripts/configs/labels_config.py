# Labels-baseline default configuration.
#
# MODEL_ARGS are passed to ``wsi.LabelsBaseline(adata, **MODEL_ARGS)``.
# TRAIN_ARGS / PLAN_KWARGS are passed to ``model.train(**TRAIN_ARGS, plan_kwargs=PLAN_KWARGS)``
# (``datasplitter_kwargs['external_indexing']`` is filled in at runtime with the splits).
#
# The baseline is a label-conditioned decoder (no encoder, no adversary), so it trains with
# scvi's default TrainingPlan and a plain NB reconstruction loss; early stopping monitors
# ``elbo_validation`` (== reconstruction loss here, since there is no KL term).

MODEL_ARGS = {
    "n_hidden": 128,
    "n_layers": 2,
    "gene_likelihood": "nb",
}

TRAIN_ARGS = {
    "max_epochs": 100,
    "batch_size": 2048,
    "check_val_every_n_epoch": 1,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_monitor": "elbo_validation",
    "enable_checkpointing": True,
    "devices": [0],
}

PLAN_KWARGS = {
    "lr": 1e-3,
}

DO_COUNTERFACTUAL = True
