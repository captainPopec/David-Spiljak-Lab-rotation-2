"""An H&E-embedding-conditioned counterpart to :class:`wsi.labels_baseline.LabelsBaseline`.

``EmbeddingsBaseline`` is deliberately the *same model* as ``LabelsBaseline`` with exactly one
thing changed: the block of the decoder input that describes **where the cell is**.

    LabelsBaseline      cond = [ onehot(cell_type) | onehot(domain)      ]   7 + 5 dims
    EmbeddingsBaseline  cond = [ onehot(cell_type) | virchow2 embedding  ]   7 + 2560 dims

Same :class:`~scvi.nn.DecoderSCVI`, same negative-binomial likelihood, same observed library
size, no encoder in either. So a difference in the LOO metrics is attributable to the domain
representation — a 5-way categorical *word* versus a continuous H&E foundation-model vector —
and not to model capacity elsewhere, optimiser settings, or the split.

The counterfactual
------------------
``LabelsBaseline``'s counterfactual is a one-line relabel: set the domain of the control cells
to ``"Tumor"`` and decode. There is no ``"Tumor"`` token here — the model was trained with 2,560
real numbers in that slot — so "pretend you are in Tumor" has to be expressed as an actual
embedding. :meth:`get_counterfactual_expression` therefore takes ``donor_indices``: for each
control cell it substitutes a **real Tumor cell's H&E vector** (that cell's embedding only; never
its expression, never its labels). Which cells may donate is the caller's decision and it changes
what the result means — donors from the scored region test "H&E describes *this* niche", donors
from elsewhere test "H&E describes tumour in general". See ``notebooks/OVCA_03_loo.ipynb``.

Two properties of that substitution are worth stating plainly rather than discovering later:

1. **Training pairs each cell with its own tile; inference pairs a cell with a stranger's.** That
   pairing is never seen during training. It is tolerable here because a 112um tile is mostly a
   description of surrounding tissue, not of the cell at its centre — a logistic regression on
   these embeddings recovers a cell's *own* type at only 32% balanced accuracy (notebook 05).
2. **The embedding must be pre-scaled by the caller.** Virchow2 features have per-feature std
   spanning 0.027 to 16, and ``DecoderSCVI``'s first layer is a plain ``Linear`` (batch-norm is
   applied *after* the multiply, not to the input), so unscaled features would enter with a 600x
   scale spread. Fit a ``StandardScaler`` on the training cells only and register the scaled
   array. Doing it outside the model keeps the fit auditable and out of the held-out ROIs.

The public API mirrors ``LabelsBaseline`` / ``cellina.Cellina`` (``setup_anndata`` / ``train`` /
``get_normalized_expression`` / ``get_counterfactual_expression``) so it drops into the
``scripts/eval.py`` LOO harness as just another method.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, ObsmField
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.nn import DecoderSCVI
from scvi.utils import setup_anndata_dsp

logger = logging.getLogger(__name__)

# Our own registry key for the per-cell H&E embedding. Summary stats expose the column count as
# ``n_embedding`` (count_stat_key = f"n_{registry_key}").
EMBEDDING_KEY = "embedding"


class EmbeddingsBaselineModule(BaseModuleClass):
    """Cell-type + H&E-embedding conditioned decoder module (no encoder).

    Parameters
    ----------
    n_input
        Number of genes.
    n_labels
        Number of cell-type categories.
    n_embedding
        Dimensionality of the per-cell H&E embedding (2,560 for Virchow2).
    n_batch
        Number of batches (0 disables batch injection).
    n_hidden
        Nodes per hidden decoder layer.
    n_layers
        Number of hidden decoder layers.
    dropout_rate
        Decoder dropout rate.
    gene_likelihood
        ``"nb"`` or ``"zinb"``.
    use_observed_lib_size
        If ``True`` (default), use ``log(x.sum(1))`` as the library size.
    """

    def __init__(
        self,
        n_input: int,
        n_labels: int,
        n_embedding: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "nb",
        use_observed_lib_size: bool = True,
    ):
        super().__init__()
        if n_labels is None or n_labels < 1:
            raise ValueError("EmbeddingsBaseline requires labels_key in setup_anndata().")
        if n_embedding is None or n_embedding < 1:
            raise ValueError("EmbeddingsBaseline requires embedding_obsm_key in setup_anndata().")

        self.n_labels = n_labels
        self.n_embedding = n_embedding
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        self.use_observed_lib_size = use_observed_lib_size
        # required by some scvi mixins / plans
        self.latent_distribution = "normal"

        # gene-level dispersion (learnable), mirroring cellina and LabelsBaseline
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        cat_list = [n_batch] if n_batch > 0 else None

        # Decoder: [onehot(labels) , embedding] -> counts. Identical to LabelsBaseline's decoder
        # apart from the width of the second input block.
        self.decoder = DecoderSCVI(
            n_labels + n_embedding,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=True,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    def _get_inference_input(self, tensors):
        return dict(
            x=tensors[REGISTRY_KEYS.X_KEY],
            labels=tensors[REGISTRY_KEYS.LABELS_KEY],
            embedding=tensors[EMBEDDING_KEY],
        )

    def _get_generative_input(self, tensors, inference_outputs):
        return dict(
            cond=inference_outputs["cond"],
            library=inference_outputs["library"],
            batch_index=tensors[REGISTRY_KEYS.BATCH_KEY],
        )

    @auto_move_data
    def inference(self, x, labels, embedding):
        """Build the conditioning vector and the observed library size.

        There is no encoder: "inference" is the cell-type one-hot concatenated with the H&E
        embedding, which is passed through unchanged (scaling is the caller's job — see the
        module docstring).
        """
        library = torch.log(x.sum(1)).unsqueeze(1)

        labels_oh = F.one_hot(labels.reshape(-1).long(), self.n_labels).float()
        cond = torch.cat([labels_oh, embedding.float()], dim=-1)

        return dict(cond=cond, library=library)

    @auto_move_data
    def generative(self, cond, library, batch_index):
        """Decode the conditioning vector into expression."""
        px_scale, _, px_rate, px_dropout = self.decoder("gene", cond, library, batch_index)
        px_r = torch.exp(self.px_r)
        return dict(px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Negative-binomial reconstruction loss only (no latent, no KL)."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        if self.gene_likelihood == "zinb":
            reconst_loss = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)

        loss = torch.mean(reconst_loss)
        # kl_local = 0 so elbo == reconstruction loss (keeps "elbo_validation" available for
        # early stopping, exactly as in LabelsBaseline).
        kl_local = torch.zeros_like(reconst_loss)
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local)

    @torch.no_grad()
    def sample(self, tensors, n_samples=1, library_size=1):
        """Sample counts from the posterior predictive distribution."""
        _, generative_outputs = self.forward(tensors, compute_loss=False)
        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]
        if self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
        else:
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        exprs = dist.sample().permute([1, 2, 0]) if n_samples > 1 else dist.sample()
        return exprs.cpu()


class EmbeddingsBaseline(UnsupervisedTrainingMixin, BaseModelClass):
    """Cell-type + H&E-embedding conditioned model.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = StandardScaler().fit(adata.obsm["virchow2"][train_idx])
    >>> adata.obsm["virchow2_scaled"] = scaler.transform(adata.obsm["virchow2"])
    >>> EmbeddingsBaseline.setup_anndata(
    ...     adata, labels_key="cell_type", embedding_obsm_key="virchow2_scaled", layer="counts"
    ... )
    >>> model = EmbeddingsBaseline(adata)
    >>> model.train()
    >>> ctrl = model.get_normalized_expression(adata[mask_control], library_size=1e4)
    >>> cf = model.get_counterfactual_expression(idx_control, donor_indices=idx_donors)
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_layers: int = 2,
        gene_likelihood: str = "nb",
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = EmbeddingsBaselineModule(
            n_input=self.summary_stats["n_vars"],
            n_labels=self.summary_stats.get("n_labels"),
            n_embedding=self.summary_stats.get(f"n_{EMBEDDING_KEY}"),
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"EmbeddingsBaseline: cell-type + H&E-embedding conditioned decoder "
            f"(n_labels={self.summary_stats.get('n_labels')}, "
            f"n_embedding={self.summary_stats.get(f'n_{EMBEDDING_KEY}')}, "
            f"{gene_likelihood} likelihood)"
        )
        self.init_params_ = self._get_init_params(locals())

        # original .obsm key for the embedding field — needed to swap in donor vectors.
        self._embedding_obsm_key = self.adata_manager.data_registry[EMBEDDING_KEY].attr_key

        logger.info("EmbeddingsBaseline model initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: Optional[str] = None,
        embedding_obsm_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        embedding_obsm_key
            Key in ``adata.obsm`` holding the per-cell H&E embedding, ``[n_cells, n_embedding]``.
            **Scale it yourself first** (``StandardScaler`` fit on the training cells only) — see
            the module docstring.
        %(param_batch_key)s
        %(param_layer)s
        """
        if embedding_obsm_key is None:
            raise ValueError("EmbeddingsBaseline requires embedding_obsm_key in setup_anndata().")

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            ObsmField(EMBEDDING_KEY, embedding_obsm_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list] = None,
        batch_size: Optional[int] = None,
        return_numpy: bool = True,
        library_size: Union[float, str] = 1.0,
    ):
        """Return normalized expression (mirrors ``LabelsBaseline.get_normalized_expression``).

        Each cell is decoded from its own cell type and its own H&E embedding. Passing the
        *target* cells here is the "direct" ceiling: the model reads the H&E of the very cells
        being predicted, so it is not a counterfactual and must not be compared as a peer.

        Parameters
        ----------
        library_size
            - float (e.g. 1e4): multiplies px_scale by this constant
            - 1: returns px_scale (pure proportions)
            - "latent": uses the observed library size (returns px_rate)
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        exprs = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = self.module.generative(**generative_inputs)

            px_scale = generative_outputs["px_scale"]
            if library_size == "latent":
                lib = torch.exp(inference_outputs["library"])
                px = px_scale * lib
            else:
                px = px_scale * library_size
            exprs.append(px.cpu())

        exprs = torch.cat(exprs, dim=0)
        return exprs.numpy() if return_numpy else exprs

    def get_counterfactual_expression(
        self,
        indices: np.ndarray,
        donor_indices: np.ndarray,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        library_size: Union[float, str] = "latent",
        return_numpy: bool = True,
    ) -> np.ndarray:
        """Predict expression for cells given someone else's H&E context.

        Copies the cells at ``indices``, overwrites each one's H&E embedding with the embedding
        of the corresponding cell in ``donor_indices`` (keeping its own cell type and observed
        library size), and decodes. This is the embedding-space analogue of ``LabelsBaseline``'s
        domain relabel: the donor's *vector* is the statement "you are now in that tissue".

        Only the donor's embedding is read — never its expression, never its labels.

        Parameters
        ----------
        indices
            Cell indices to compute the counterfactual for (the control arm).
        donor_indices
            Same length as ``indices``. ``donor_indices[i]`` is the cell whose H&E embedding is
            handed to ``indices[i]``. Sample these from the target domain; excluding the held-out
            cell type is the caller's responsibility.
        library_size
            Passed to :meth:`get_normalized_expression`; defaults to ``"latent"`` (observed),
            matching ``LabelsBaseline``.

        Returns
        -------
        np.ndarray of shape ``(len(indices), n_genes)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        indices = np.asarray(indices)
        donor_indices = np.asarray(donor_indices)
        if donor_indices.shape[0] != indices.shape[0]:
            raise ValueError(
                f"donor_indices has length {donor_indices.shape[0]} but indices has "
                f"length {indices.shape[0]}; one donor per cell is required."
            )

        adata_cf = adata[indices].copy()
        donor_embeddings = np.asarray(adata.obsm[self._embedding_obsm_key])[donor_indices]
        adata_cf.obsm[self._embedding_obsm_key] = np.ascontiguousarray(donor_embeddings)

        return self.get_normalized_expression(
            adata=adata_cf, indices=None, batch_size=batch_size,
            library_size=library_size, return_numpy=return_numpy,
        )
