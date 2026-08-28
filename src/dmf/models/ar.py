"""Multivariate least-squares AR(p), fitted per training split.

Direct multi-horizon: one coefficient matrix per horizon step, so that a forecast at
``h = 30`` is a single linear map from the lookback rather than thirty compounding
one-step predictions. Fitted on CPU with NumPy; there is no training loop.

Gate 3 uses this model as a difficulty check on the task itself. If AR(p) already reaches
0.8 skill at a 3 s horizon on roll, the task is too easy as specified and the horizon or
observation mode must change before any effort goes into deep models.

**Feature ordering is load-bearing.** ``lag_features(x, p)`` is lag-major and
most-recent-lag-first: block ``k`` holds all ``C_in`` channels at lag ``k`` samples back
from the end of the lookback. That makes the orders **prefix-nested** --
``lag_features(x, 10) == lag_features(x, 40)[:, :10*C_in]`` -- so a single accumulation of
the normal equations at ``p = 40`` yields the exact normal equations for ``p = 10`` and
``p = 20`` as leading sub-blocks. The whole one-pass three-order fit rests on that
property, so it is asserted in ``tests/test_models.py`` rather than described here.

The same layout also makes the *channel* subsets free. Within lag block ``k`` the channels
appear in input order, so the design matrix of the first ``m`` input channels is the
column subset ``{k * C_in + c : k < p, c < m}`` of the six-channel one -- strided rather
than a prefix, but still a plain slice of the already-accumulated Gram and cross moments.
That is what lets ``ar_attitude_only`` (``n_input_used = 3``, the attitudes under the P2-D4
prefix rule, at order 40 so that its 120 features match ``ar20``'s 20 x 6 exactly) be
fitted from the *same* moments pass as ``ar20`` and ``ar40``, with no second pass over the
training split. :func:`dmf.train.closed_form.subset_columns` builds the index and
``tests/test_models.py`` checks the sliced fit against a design matrix built directly on
three channels.
"""

import numpy as np
import torch
from torch import Tensor

from dmf.models.base import BaseForecaster
from dmf.train.registry import register_model
from dmf.typedefs import FloatArray

__all__ = ["ARForecaster", "lag_features"]


def lag_features(x: Tensor, order: int) -> Tensor:
    """Build the lag-major, most-recent-first design block of an AR(p) regression.

    Args:
        x: Input windows, shape ``(B, L, C_in)``, dimensionless.
        order: AR order ``p``, samples of history, in ``[1, L]``.

    Returns:
        Design block, shape ``(B, p * C_in)``. Column ``k * C_in + c`` is channel ``c`` at
        lag ``k`` samples back from the last lookback sample, so lag 0 (the most recent
        sample) occupies the leading ``C_in`` columns.

    Raises:
        ValueError: If ``order`` is not in ``[1, L]`` or ``x`` is not three-dimensional.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, L, C_in), got {tuple(x.shape)}")
    lookback = int(x.shape[1])
    if not 1 <= order <= lookback:
        raise ValueError(f"order must be in [1, {lookback}], got {order}")
    return x[:, lookback - order :, :].flip(dims=[1]).reshape(x.shape[0], order * int(x.shape[2]))


@register_model("ar")
class ARForecaster(BaseForecaster):
    """Least-squares multivariate autoregressive forecaster.

    Attributes are set by :meth:`fit`; the model is not usable before that call.

    Coefficients live in **buffers**, not ``nn.Parameter``: nothing differentiates through
    them, and holding them as parameters would invite an optimiser to touch a closed-form
    solution. The cost is that ``sum(p.numel() for p in model.parameters())`` reports 0, so
    :attr:`n_fitted_parameters` is overridden -- see ``models/base.py``.
    """

    FIT_KIND = "closed_form"

    def __init__(
        self,
        lookback: int,
        max_horizon: int,
        n_input_channels: int,
        n_target_channels: int,
        order: int = 20,
        ridge: float = 0.0,
        n_input_used: int | None = None,
    ) -> None:
        """Configure the AR model.

        Args:
            lookback: Input window length ``L``, samples. Must be at least ``order``.
            max_horizon: Forecast length ``H``, samples.
            n_input_channels: Input channel count ``C_in``, i.e. the width of the tensor
                ``forward`` is handed. Unchanged by ``n_input_used``.
            n_target_channels: Target channel count ``C_out``.
            order: AR order ``p``, samples of history used per prediction. The sweep in
                Phase 3 covers ``p`` in {10, 20, 40}.
            ridge: Tikhonov regularisation strength, dimensionless. Nonzero values guard
                against the ill-conditioning that a narrowband signal produces in the
                lagged design matrix. Dimensionless because the solve is centred and
                whitened before the penalty is applied.
            n_input_used: How many of the ``C_in`` input channels the regression may read,
                counting from the front, or None for all of them. Under the P2-D4 prefix
                rule the first three are roll, pitch and heave, so ``n_input_used = 3`` is
                the attitude-only information set. The **target** set is untouched: the
                model still forecasts all ``C_out`` channels. This is an information-set
                ablation, not a smaller task, and it is deliberately the only thing that
                differs between ``ar20`` and ``ar_attitude_only`` -- which is why the
                ablation runs at ``order = 40``: ``40 * 3`` features is exactly ``20 * 6``,
                so the pair is matched on capacity as well as on solver and ridge.

        Raises:
            ValueError: If ``order`` exceeds ``lookback`` or is not positive, if ``ridge``
                is negative, or if ``n_input_used`` is outside ``[1, C_in]``.
        """
        super().__init__(lookback, max_horizon, n_input_channels, n_target_channels)
        if not 1 <= order <= lookback:
            raise ValueError(f"order must be in [1, lookback={lookback}], got {order}")
        if ridge < 0.0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")
        used = n_input_channels if n_input_used is None else int(n_input_used)
        if not 1 <= used <= n_input_channels:
            raise ValueError(
                f"n_input_used must be in [1, n_input_channels={n_input_channels}], "
                f"got {n_input_used}"
            )
        self.order = order
        self.ridge = ridge
        self.n_input_used = used
        n_features = order * used
        n_outputs = max_horizon * n_target_channels
        self.register_buffer("weight", torch.zeros(n_features, n_outputs))
        self.register_buffer("bias", torch.zeros(n_outputs))
        self._fitted = False

    @property
    def n_fitted_parameters(self) -> int:
        """Coefficient count: ``p * n_input_used * H * C_out`` plus one intercept per output.

        216 900 at ``p = 40`` on all six inputs with the P3 geometry (``L = 200``,
        ``H = 150``, ``C_in = C_out = 6``); 108 900 for ``ar20``, and 108 900 again for
        ``ar_attitude_only``, which reads three of the six input channels at twice the
        order. That equality is the point of the pair and is asserted in
        ``tests/test_models.py``. Reported instead of the parameter count because the
        coefficients are buffers.
        """
        if not self._fitted:
            return 0
        return int(self.weight.numel() + self.bias.numel())

    def set_coefficients(self, weight: FloatArray, bias: FloatArray) -> None:
        """Install a solved coefficient matrix.

        Args:
            weight: Coefficients, shape ``(p * n_input_used, H * C_out)``, dimensionless.
                The output axis is ``H``-major: column ``h * C_out + c``.
            bias: Intercepts, shape ``(H * C_out,)``, dimensionless.

        Raises:
            ValueError: If either array has the wrong shape.
        """
        expected_w = (self.order * self.n_input_used, self.max_horizon * self.n_target_channels)
        w = np.asarray(weight, dtype=np.float64)
        b = np.asarray(bias, dtype=np.float64)
        if w.shape != expected_w:
            raise ValueError(f"weight has shape {w.shape}, expected {expected_w}")
        if b.shape != (expected_w[1],):
            raise ValueError(f"bias has shape {b.shape}, expected {(expected_w[1],)}")
        self.weight = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
        self.bias = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
        self._fitted = True

    def fit(self, x: FloatArray, y: FloatArray) -> None:
        """Solve for one coefficient matrix per horizon step by least squares.

        Must be called with **training-split windows only**.

        Array form, kept because the docstring contract and the unit tests use it. It
        forms the moments and delegates to
        :func:`dmf.train.closed_form.fit_ar_from_moments`; production fitting streams the
        same moments over a dataloader instead, because ``x`` for
        ``unseen_vessel/train`` is ``(1_878_432, 200, 6)``, i.e. 18 GB in float64.

        Args:
            x: Training input windows, shape ``(N, L, C_in)``, dimensionless.
            y: Training target windows, shape ``(N, H, C_out)``, dimensionless -- i.e.
                already put through :func:`dmf.data.normalize.normalize_target`.

        Raises:
            ValueError: If ``x`` and ``y`` disagree on ``N``, or if ``N`` is smaller than
                the number of free parameters per horizon step.
        """
        from dmf.train.closed_form import LagMoments, solve_ar_coefficients

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(
                f"x must be (N, L, C_in) and y (N, H, C_out), got {x.shape}, {y.shape}"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x has {x.shape[0]} windows but y has {y.shape[0]}")
        n_features = self.order * self.n_input_used
        if x.shape[0] < n_features + 1:
            raise ValueError(
                f"{x.shape[0]} windows is fewer than the {n_features + 1} free parameters "
                f"per horizon step at order {self.order}; the normal equations are singular"
            )
        # Moments are formed over **all** ``C_in`` channels even when only the first
        # ``n_input_used`` are read, so that this array path takes the same channel-subset
        # slice inside ``solve_ar_coefficients`` that the streaming production path takes.
        # Accumulating the narrow design matrix here instead would leave that slice
        # untested by every test that goes through ``fit``.
        feats = lag_features(torch.from_numpy(x), self.order).numpy()
        targets = y.reshape(y.shape[0], -1)
        moments = LagMoments(
            n=int(x.shape[0]),
            sx=feats.sum(axis=0),
            sy=targets.sum(axis=0),
            syy=np.square(targets).sum(axis=0),
            gram=feats.T @ feats,
            cross=feats.T @ targets,
            max_order=self.order,
            n_input_channels=self.n_input_channels,
            max_horizon=self.max_horizon,
            n_target_channels=self.n_target_channels,
        )
        weight, bias, _ = solve_ar_coefficients(
            moments, order=self.order, ridge=self.ridge, n_input_used=self.n_input_used
        )
        self.set_coefficients(weight, bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the fitted coefficient matrices.

        The **full** ``(B, L, C_in)`` window is accepted whatever ``n_input_used`` is; the
        channel subset is taken here, exactly as :class:`dmf.models.persistence.Persistence`
        slices the first ``C_out`` inputs. That keeps every AR variant a drop-in
        ``ForecastModel`` with one input signature, so neither
        :func:`dmf.train.experiment.run_experiment` nor the ONNX export needs a special
        case for the ablation.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Point forecasts, shape ``(B, H, C_out)``, dimensionless.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "ARForecaster has no coefficients; call fit or dmf.train.closed_form.fit_ar first"
            )
        feats = lag_features(x[:, :, : self.n_input_used], self.order)
        flat = feats @ self.weight.to(dtype=x.dtype) + self.bias.to(dtype=x.dtype)
        return flat.view(x.shape[0], self.max_horizon, self.n_target_channels)
