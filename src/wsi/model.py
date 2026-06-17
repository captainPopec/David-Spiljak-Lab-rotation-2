"""Scaffold for the new WSI + ``cellina`` method.

The goal of this project is to bring **whole-slide-image (WSI)** information into the
``cellina`` spatial-perturbation framework. ``cellina`` already separates a cell's
intrinsic state ``z`` from its spatial context ``s`` (the influence of its transcriptomic
neighbours). The idea here is to additionally condition on, or fuse in, features derived
from the H&E / WSI image of the same tissue (e.g. via LazySlide tile embeddings).

Nothing is implemented yet — this is where your method goes. A sketch of the intended
shape is below; adapt freely.

See also
--------
- ``notebooks/loo_eval.ipynb``     : the cellina-vs-CPA leave-one-out benchmark your
                                     method will be evaluated against.
- ``notebooks/01_spatialdata_load.ipynb`` : loading the CRC data (+ image) as SpatialData.
- ``notebooks/02_lazyslide.ipynb`` : extracting WSI tile features with LazySlide.
- ``scripts/metrics.py``           : the evaluation metrics (``compare_adatas``).
"""

from __future__ import annotations


class WSICellina:
    """Placeholder for the WSI-conditioned cellina model.

    TODO:
      - Decide how WSI features enter the model: as an extra conditioning input to the
        spatial encoder (``s``), as an auxiliary reconstruction target, or as a separate
        fused latent.
      - Wrap or subclass ``cellina.Cellina`` rather than reimplementing the VAE.
      - Expose a counterfactual API mirroring
        ``cellina.Cellina.get_counterfactual_expression`` so it slots into the existing
        ``scripts/eval.py`` / ``notebooks/loo_eval.ipynb`` harness unchanged.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "WSICellina is a scaffold. Implement your WSI + cellina method here. "
            "See src/wsi/README.md for the plan."
        )
