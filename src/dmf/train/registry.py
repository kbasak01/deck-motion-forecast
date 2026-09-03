"""Model registry.

Adding a model means adding one file under ``src/dmf/models/``, one config under
``configs/model/``, and one registry entry here. Nothing else changes -- that constraint is
what keeps the model set open to a sixth or seventh entry without a refactor.

The registry is populated by import side effect, so it is only correct if
``dmf.models`` has been imported. :func:`build_model` imports it explicitly rather than
relying on the caller having done so: an empty registry caused by import order is a bug
that presents as "unknown model", which reads like a typo in a config file and sends the
reader to the wrong place.
"""

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

from dmf.config import ModelConfig
from dmf.data.windows import WindowSpec

if TYPE_CHECKING:  # pragma: no cover - import cycle: models import this module to register
    from dmf.models.base import BaseForecaster

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]

#: Registry key -> model class. Populated by the :func:`register_model` decorator at
#: import time.
MODEL_REGISTRY: dict[str, "type[BaseForecaster]"] = {}


def register_model(name: str) -> "Callable[[type[BaseForecaster]], type[BaseForecaster]]":
    """Return a decorator that registers a model class under ``name``.

    Args:
        name: Registry key, matching the ``name`` field of a ``configs/model/*.yaml``
            file.

    Returns:
        A class decorator that inserts the class into :data:`MODEL_REGISTRY` and returns
        it unchanged.

    Raises:
        ValueError: If ``name`` is already registered. Silent overwrites would make the
            model that actually ran depend on import order.
    """

    def decorator(cls: "type[BaseForecaster]") -> "type[BaseForecaster]":
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"model {name!r} is already registered to "
                f"{MODEL_REGISTRY[name].__module__}.{MODEL_REGISTRY[name].__qualname__}; "
                f"a silent overwrite would make the model that ran depend on import order"
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(
    cfg: ModelConfig,
    window_spec: WindowSpec,
    n_input_channels: int,
    n_target_channels: int,
) -> "BaseForecaster":
    """Instantiate the model named by a config.

    Args:
        cfg: Model configuration, supplying the registry key, head type, quantile levels,
            and architecture keyword arguments.
        window_spec: Window geometry, supplying ``lookback`` and ``max_horizon`` in
            samples.
        n_input_channels: Input channel count ``C_in``.
        n_target_channels: Target channel count ``C_out``.

    Returns:
        The constructed model, on CPU and untrained.

    Raises:
        KeyError: If ``cfg.name`` is not registered.
        TypeError: If ``cfg.params`` contains a key the model constructor does not accept,
            or if the model cannot carry the head its config asks for -- a baseline has no
            ``head`` argument, so ``head: gaussian`` on ``persistence`` raises here instead
            of quietly building a point model.
        ValueError: If a quantile config's levels are not the fan a model of that width
            reports.
    """
    import dmf.models  # noqa: F401  (import side effect populates MODEL_REGISTRY)

    # Deferred for the same reason as the line above: every model module imports this one
    # to register itself, so a module-level import of anything under `dmf.models` here
    # would close the cycle.
    from dmf.models.heads import quantile_fan

    if cfg.name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {cfg.name!r}; registered models are {sorted(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[cfg.name]

    kwargs: dict[str, object] = dict(cfg.params)
    # Passed only for a non-point config, so every pre-Phase-5 call is byte-for-byte the
    # call it was. It also closes a hole: `head: gaussian` used to fall through this
    # function silently and build a POINT model under a config that documents a Gaussian
    # one. With `head` in `kwargs`, the accepted-set check below raises on any model that
    # cannot carry the head its config asks for.
    if cfg.head != "point":
        kwargs["head"] = cfg.head
    if cfg.head == "quantile":
        kwargs["n_quantiles"] = len(cfg.quantiles)
        # A model is constructed with a fan *width* and recovers the levels from
        # `dmf.models.heads.quantile_fan`, so a config declaring different levels would be
        # trained on its own fan (`resolve_loss` reads `cfg.quantiles`) and scored on the
        # project fan. Loud rather than silent.
        expected = quantile_fan(len(cfg.quantiles))
        if any(abs(a - b) > 1e-9 for a, b in zip(cfg.quantiles, expected, strict=True)):
            raise ValueError(
                f"model {cfg.label or cfg.name!r} declares quantile levels "
                f"{list(cfg.quantiles)}, but a model of fan width {len(cfg.quantiles)} "
                f"reports {list(expected)}; the levels a model is scored at come from "
                f"dmf.models.heads.quantile_fan, so the two must agree"
            )

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise TypeError(
            f"model {cfg.name!r} ({cls.__qualname__}) does not accept {unknown}; "
            f"accepted keyword arguments are {sorted(accepted)}"
        )
    return cls(
        lookback=window_spec.lookback,
        max_horizon=window_spec.max_horizon,
        n_input_channels=n_input_channels,
        n_target_channels=n_target_channels,
        **kwargs,  # type: ignore[arg-type]
    )
