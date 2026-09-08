"""Forecasting models, all behind one interface.

Every model in this subpackage implements the :class:`dmf.models.base.ForecastModel`
protocol: ``forward(x: Tensor[B, L, C_in]) -> Tensor[B, H, C_out]`` for point models, or
``-> Tensor[B, H, C_out, Q]`` for quantile models. Output is **direct multi-horizon** --
all ``H`` steps are emitted in one pass, never by autoregressive rollout, which avoids
error compounding and keeps the exported ONNX graph a single fixed shape.

Adding a model means adding one file here, one config under ``configs/model/``, and one
registry entry. Nothing else changes.

**Subclasses implement ``_predict``, not ``forward``.**
:meth:`dmf.models.base.BaseForecaster.forward` is a template method: it applies the optional
reversible instance normalisation (``DataConfig.revin``) on the way in, calls ``_predict``,
and inverts it on the way out. A model that defines its own ``forward`` still builds, trains
and scores -- it simply ignores its arm's ``revin: true``, which appears in no column of any
table, so ``tests/test_models.py`` asserts that none does.

The line-up deliberately includes models that may beat the deep ones. DLinear -- a single
linear layer on a decomposed window -- is notoriously competitive with transformers on
long-horizon forecasting. If it wins here, that is a result to report, not a failure to
hide.
"""

from dmf.models.ar import ARForecaster
from dmf.models.base import BaseForecaster, ForecastModel, QuantileForecastModel
from dmf.models.dlinear import DLinear
from dmf.models.dlinear_ols import DLinearOLS
from dmf.models.lstm import LSTMForecaster
from dmf.models.persistence import DampedPersistence, Persistence
from dmf.models.residual_interval import EmpiricalResidualInterval
from dmf.models.tcn import TCN, TemporalBlock, receptive_field
from dmf.models.transformer import PatchEmbedding, TransformerForecaster
from dmf.models.window_mean import WindowMean

__all__ = [
    "TCN",
    "ARForecaster",
    "BaseForecaster",
    "DLinear",
    "DLinearOLS",
    "DampedPersistence",
    "EmpiricalResidualInterval",
    "ForecastModel",
    "LSTMForecaster",
    "PatchEmbedding",
    "Persistence",
    "QuantileForecastModel",
    "TemporalBlock",
    "TransformerForecaster",
    "WindowMean",
    "receptive_field",
]
