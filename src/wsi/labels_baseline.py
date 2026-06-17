"""A simple label-conditioned baseline for the cellina LOO benchmark.

``LabelsBaseline`` ignores expression and spatial context entirely. It predicts gene
expression purely from two categorical labels — the cell type (``labels_key``, e.g.
``coarse_type``) and the spatial domain (``domains_key``, e.g. ``typ_clean``) — which are
one-hot encoded and decoded through a standard scvi-tools :class:`~scvi.nn.DecoderSCVI`
head with a negative-binomial likelihood and observed library size.

Cell type and domain enter as **two separate one-hot blocks** concatenated into the
decoder input (``n_input = n_labels + n_domains``). Because the decoder shares parameters
across these blocks, it can produce a prediction for a ``(cell_type, domain)`` combination
that was never observed during training (e.g. held-out ``(Myeloid, CRC)``) — composing a
learned "cell type" effect with a learned "domain" effect. A literal groupby-mean could
not do this, which is why a decoder is used.

The public API mirrors :class:`cellina.Cellina` (``setup_anndata`` / ``train`` /
``get_normalized_expression`` / ``get_counterfactual_expression``) so it slots into the
``scripts/eval.py`` LOO harness as just another method. Unlike cellina's spatial
counterfactual, the baseline counterfactual simply swaps the domain label (CPA semantics):
take control cells, relabel their domain to the target, and decode.
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
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin
from scvi.nn import DecoderSCVI
from scvi.utils import setup_anndata_dsp

logger = logging.getLogger(__name__)

# Our own registry key for the spatial-domain categorical field. Summary stats expose it
# as ``n_domains`` (count_stat_key = f"n_{registry_key}").
DOMAINS_KEY = "domains"


class LabelsBaselineModule(BaseModuleClass):
    """Label-conditioned decoder module (no encoder).

    Parameters
    ----------
    n_input
        Number of genes.
    n_labels
        Number of cell-type categories.
    n_domains
        Number of spatial-domain categories.
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
        n_domains: int,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        gene_likelihood: str = "nb",
        use_observed_lib_size: bool = True,
    ):
        super().__init__()
        if n_labels is None or n_labels < 1:
            raise ValueError("LabelsBaseline requires labels_key in setup_anndata().")
        if n_domains is None or n_domains < 1:
            raise ValueError("LabelsBaseline requires domains_key in setup_anndata().")

        self.n_labels = n_labels
        self.n_domains = n_domains
        self.n_batch = n_batch
        self.gene_likelihood = gene_likelihood
        self.use_observed_lib_size = use_observed_lib_size
        # required by some scvi mixins / plans
        self.latent_distribution = "normal"

        # gene-level dispersion (learnable), mirroring cellina
        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        cat_list = [n_batch] if n_batch > 0 else None

        # Decoder: [onehot(labels) , onehot(domains)] -> counts
        self.decoder = DecoderSCVI(
            n_labels + n_domains,
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
            domains=tensors[DOMAINS_KEY],
        )

    def _get_generative_input(self, tensors, inference_outputs):
        return dict(
            cond=inference_outputs["cond"],
            library=inference_outputs["library"],
            batch_index=tensors[REGISTRY_KEYS.BATCH_KEY],
        )

    @auto_move_data
    def inference(self, x, labels, domains):
        """Build the one-hot conditioning vector and the observed library size.

        There is no encoder: "inference" is just the labels.
        """
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        else:  # keep the option open, but default path is observed
            library = torch.log(x.sum(1)).unsqueeze(1)

        labels_oh = F.one_hot(labels.reshape(-1).long(), self.n_labels).float()
        domains_oh = F.one_hot(domains.reshape(-1).long(), self.n_domains).float()
        cond = torch.cat([labels_oh, domains_oh], dim=-1)

        return dict(cond=cond, library=library)

    @auto_move_data
    def generative(self, cond, library, batch_index):
        """Decode the one-hot labels into expression."""
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
        # kl_local = 0 so elbo == reconstruction loss (keeps "elbo_validation" available
        # for early stopping).
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


class LabelsBaseline(UnsupervisedTrainingMixin, BaseModelClass):
    """Label-conditioned baseline model.

    Examples
    --------
    >>> LabelsBaseline.setup_anndata(adata, labels_key="coarse_type",
    ...                              domains_key="typ_clean", layer="counts")
    >>> model = LabelsBaseline(adata)
    >>> model.train()
    >>> ctrl = model.get_normalized_expression(adata[mask], library_size=1e4)
    >>> cf = model.get_counterfactual_expression(idx_control, target_domain="CRC")
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

        self.module = LabelsBaselineModule(
            n_input=self.summary_stats["n_vars"],
            n_labels=self.summary_stats.get("n_labels"),
            n_domains=self.summary_stats.get("n_domains"),
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_layers=n_layers,
            gene_likelihood=gene_likelihood,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"LabelsBaseline: label-conditioned decoder "
            f"(n_labels={self.summary_stats.get('n_labels')}, "
            f"n_domains={self.summary_stats.get('n_domains')}, {gene_likelihood} likelihood)"
        )
        self.init_params_ = self._get_init_params(locals())

        # original .obs column for the domain field — needed for counterfactual relabelling
        self._domains_obs_key = self.adata_manager.get_state_registry(DOMAINS_KEY).original_key

        logger.info("LabelsBaseline model initialized")

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: Optional[str] = None,
        domains_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        layer: Optional[str] = None,
        **kwargs,
    ) -> Optional[AnnData]:
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_labels_key)s
        domains_key
            Key in ``adata.obs`` for the spatial-domain labels (categorical).
        %(param_batch_key)s
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(DOMAINS_KEY, domains_key),
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
        """Return normalized expression (mirrors :meth:`cellina.Cellina.get_normalized_expression`).

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
        target_domain: str,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        library_size: Union[float, str] = "latent",
        return_numpy: bool = True,
    ) -> np.ndarray:
        """Predict expression for cells moved to ``target_domain``.

        Copies the cells at ``indices``, relabels their spatial domain to ``target_domain``
        (keeping cell type and observed library size), and decodes. Mirrors CPA's
        counterfactual semantics (control cells moved to the held-out domain).

        Parameters
        ----------
        indices
            Cell indices to compute the counterfactual for.
        target_domain
            Domain category to assign to every cell (must be one seen in training).
        library_size
            Passed to :meth:`get_normalized_expression`; defaults to ``"latent"`` (observed).

        Returns
        -------
        np.ndarray of shape ``(len(indices), n_genes)``.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)

        adata_cf = adata[np.asarray(indices)].copy()
        adata_cf.obs[self._domains_obs_key] = target_domain
        # re-cast to category so the field transfer maps it through the trained mapping
        adata_cf.obs[self._domains_obs_key] = adata_cf.obs[self._domains_obs_key].astype("category")

        return self.get_normalized_expression(
            adata=adata_cf, indices=None, batch_size=batch_size,
            library_size=library_size, return_numpy=return_numpy,
        )
