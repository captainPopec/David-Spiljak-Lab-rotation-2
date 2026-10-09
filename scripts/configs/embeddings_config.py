# Embeddings-baseline default configuration.
#
# MODEL_ARGS are passed to ``wsi.EmbeddingsBaseline(adata, **MODEL_ARGS)``.
# TRAIN_ARGS / PLAN_KWARGS are passed to ``model.train(**TRAIN_ARGS, plan_kwargs=PLAN_KWARGS)``
# (``datasplitter_kwargs['external_indexing']`` is filled in at runtime with the splits).
#
# EVERY VALUE HERE IS COPIED FROM labels_config.py ON PURPOSE. The point of the comparison in
# OVCA_03_loo.ipynb is that the two models differ in exactly one thing -- the domain one-hot is
# replaced by a Virchow2 embedding -- so any tuning advantage given to one of them would make a
# metric gap unattributable. If you tune one, tune both, and say so in the notebook.

EMBEDDING_KEY = "virchow2"           # obsm key in the ovca_loo bundle (2560-d)
SCALED_EMBEDDING_KEY = "virchow2_scaled"  # obsm key written by the notebook after StandardScaler

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

# Seeds for the repeated runs. Both models are retrained under each seed, with a fresh donor
# draw, and section D reports mean +- sd -- without this a pearson gap of a few hundredths is
# indistinguishable from initialisation luck on 560 OOD cells.
SEEDS = [0, 1, 2]
