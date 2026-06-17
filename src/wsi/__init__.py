"""``wsi`` — new whole-slide-image (WSI) + ``cellina`` methods.

This package is the home for the *new* methods developed in this project (as opposed
to ``scripts/``, which holds the evaluation harness and competitor baselines such as
CPA). It is currently a scaffold: see :mod:`wsi.model` and ``src/wsi/README.md`` for
what is meant to live here.
"""

__version__ = "0.0.1"

from wsi.labels_baseline import LabelsBaseline

__all__ = ["LabelsBaseline"]
