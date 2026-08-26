"""Model registry.

Adding a model means adding one file under ``src/dmf/models/``, one config under
``configs/model/``, and one registry entry here. Nothing else changes -- that constraint is
what keeps the model set open to a sixth or seventh entry without a refactor.
"""

from collections.abc import Callable

from dmf.config import ModelConfig
from dmf.data.windows import WindowSpec
from dmf.models.base import BaseForecaster

__all__ = ["MODEL_REGISTRY", "build_model", "register_model"]

#: Registry key -> model class. Populated by the :func:`register_model` decorator at
#: import time.
MODEL_REGISTRY: dict[str, type[BaseForecaster]] = {}


def register_model(name: str) -> Callable[[type[BaseForecaster]], type[BaseForecaster]]:
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
    raise NotImplementedError


def build_model(
    cfg: ModelConfig,
    window_spec: WindowSpec,
    n_input_channels: int,
    n_target_channels: int,
) -> BaseForecaster:
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
        TypeError: If ``cfg.params`` contains a key the model constructor does not accept.
    """
    raise NotImplementedError
