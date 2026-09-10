"""ONNX export.

Opset 18, **dynamic batch axis only**. The sequence length is fixed deliberately: a
dynamic sequence axis prevents the runtime from selecting specialised kernels and buys
nothing here, since the lookback is a fixed property of the trained model.

**What the graph contains that ``forward`` does not.** A quantile model's ``forward``
returns the head's raw ``(B, H, C, Q)`` output, unsorted: pinball loss does not constrain
quantile ordering, and the sort lives downstream in
:class:`dmf.models.heads.PredictiveDistribution`. Tracing the model as-is would therefore
ship a graph whose intervals cross -- the trap ``CLAUDE.md`` names -- because the deployed
consumer of an ONNX file has no ``PredictiveDistribution`` to route through. The exported
graph carries :func:`dmf.models.heads.sort_quantiles` as its last node, **only** for a
``quantile`` head. Never for ``point`` (rank-3, and the sort would raise), and never for
``gaussian``: that head's trailing axis is ``(mean, log_var)``, not a fan, and sorting it
is silent, shape-preserving and catastrophic (``docs/protocol.md`` P5-D1).
"""

from pathlib import Path

import torch
from torch import Tensor, nn

from dmf.models.base import BaseForecaster
from dmf.models.heads import sort_quantiles

__all__ = [
    "INPUT_NAME",
    "OUTPUT_NAME",
    "export_model",
    "graph_dims",
    "graph_input_geometry",
    "wrap_for_export",
]

#: Graph input name. Fixed rather than derived, because the parity check, the benchmark
#: harness and any downstream consumer all bind by name.
INPUT_NAME: str = "input"

#: Graph output name.
OUTPUT_NAME: str = "output"


class _SortedQuantileGraph(nn.Module):
    """A quantile model plus the post-hoc quantile sort, as one traceable module.

    Attributes:
        inner: The wrapped quantile forecaster.
    """

    def __init__(self, inner: BaseForecaster) -> None:
        """Wrap a quantile forecaster.

        Args:
            inner: The model to wrap. Must carry a ``quantile`` head; the caller
                (:func:`wrap_for_export`) is what enforces that.
        """
        super().__init__()
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        """Forecast and sort the quantile axis ascending.

        Args:
            x: Input windows, shape ``(B, L, C_in)``, dimensionless.

        Returns:
            Quantile forecasts, shape ``(B, H, C_out, Q)``, dimensionless, non-decreasing
            along the last axis.
        """
        return sort_quantiles(self.inner.forward(x))


def wrap_for_export(model: BaseForecaster) -> nn.Module:
    """Return the module whose forward pass the exported graph must reproduce.

    This is also the **parity reference**: :func:`dmf.deploy.parity.check_parity` compares
    ONNX against this module and not against ``model.forward``, so that the sort the graph
    carries is part of what is checked rather than a difference the tolerance has to
    absorb.

    Args:
        model: The trained model.

    Returns:
        ``model`` itself for a ``point`` or ``gaussian`` head; ``model`` wrapped in the
        quantile sort for a ``quantile`` head.
    """
    if model.head_kind == "quantile":
        return _SortedQuantileGraph(model)
    return model


def _declare_output_dims(path: Path, fixed_dims: tuple[int, ...], batch_symbol: str) -> None:
    """Re-declare the graph output's non-batch axes as the fixed extents they are.

    ``torch.sort`` exports as ``TopK``, whose ``K`` is a graph input rather than an
    attribute, so ONNX shape inference gives up on **every** trailing axis and the quantile
    graphs came out declaring ``(batch, TopKoutput_dim_1, TopKoutput_dim_2,
    TopKoutput_dim_3)``. Those axes are not dynamic -- ``H``, ``C_out`` and ``Q`` are fixed
    properties of the trained model -- and leaving them symbolic would mean the exported
    contract disagreed with the model's own
    :meth:`dmf.models.base.BaseForecaster.output_shape` for exactly the two graphs whose
    output shape is hardest to reason about, and that a consumer allocating from the graph
    could not size its buffer.

    Rewriting a declared shape is only safe because it is checked twice downstream: ORT
    validates the declaration against what the graph computes at session run, and
    :func:`dmf.deploy.parity.check_parity` asserts every produced array against
    ``output_shape`` before comparing values. A declaration that were wrong would fail
    both, loudly, rather than propagate.

    Args:
        path: The ``.onnx`` file, rewritten in place.
        fixed_dims: The non-batch output extents, in order.
        batch_symbol: Symbol name for the dynamic batch axis.

    Raises:
        ValueError: If the graph's output rank disagrees with ``fixed_dims``, or if an axis
            the exporter already resolved disagrees with the model's declared extent. Both
            mean the graph computes something other than what the model says it does, which
            is not a declaration to repair.
    """
    import onnx

    proto = onnx.load(str(path))
    dims = proto.graph.output[0].type.tensor_type.shape.dim
    if len(dims) != 1 + len(fixed_dims):
        raise ValueError(
            f"{path}: graph output is rank {len(dims)} but the model declares rank "
            f"{1 + len(fixed_dims)}"
        )
    for dim, expected in zip(dims[1:], fixed_dims, strict=True):
        if dim.dim_value and dim.dim_value != expected:
            raise ValueError(
                f"{path}: graph output declares extent {dim.dim_value} where the model "
                f"declares {expected}"
            )
    dims[0].Clear()
    dims[0].dim_param = batch_symbol
    for dim, expected in zip(dims[1:], fixed_dims, strict=True):
        dim.Clear()
        dim.dim_value = expected
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))


def export_model(
    model: BaseForecaster,
    sample_input: Tensor,
    path: Path,
    opset: int = 18,
) -> Path:
    """Export a trained model to ONNX.

    Args:
        model: The trained model. Placed in eval mode and moved to CPU before tracing, so
            that the exported graph does not carry device-specific nodes.
        sample_input: A representative input window, shape ``(B, L, C_in)``,
            dimensionless. Its ``L`` and ``C_in`` become fixed in the graph; ``B`` is made
            dynamic.
        path: Destination ``.onnx`` file. Lives under ``artifacts/`` and is gitignored.
        opset: ONNX opset version.

    Returns:
        The path written.

    Raises:
        ValueError: If ``sample_input`` does not match the model's declared ``L`` and
            ``C_in``.
    """
    if sample_input.ndim != 3:
        raise ValueError(
            f"sample_input must have shape (B, L, C_in), got {tuple(sample_input.shape)}"
        )
    _, lookback, n_channels = sample_input.shape
    if (lookback, n_channels) != (model.lookback, model.n_input_channels):
        raise ValueError(
            f"sample_input is (L={lookback}, C_in={n_channels}) but the model declares "
            f"(L={model.lookback}, C_in={model.n_input_channels}); the exported graph "
            f"fixes both axes, so a mismatched trace would ship a graph the model's own "
            f"windows do not fit"
        )
    if sample_input.shape[0] < 1:
        raise ValueError("sample_input must carry at least one window")

    model.eval()
    model.to("cpu")
    graph = wrap_for_export(model)
    graph.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            graph,
            (sample_input.detach().to("cpu").float(),),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            # Batch only. The sequence and channel axes are fixed on purpose.
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            # The TorchScript exporter, explicitly. It is deprecated in torch 2.13 in
            # favour of the torch.export-based one, which is the default -- but that path
            # requires `onnxscript`, which is not in this project's pinned dependency set,
            # and adding a dependency is not a change to make silently inside an export
            # helper. Pinning the flag also means the exporter in use does not change
            # under the benchmark when torch is upgraded, which for a latency study is the
            # more important property of the two.
            dynamo=False,
        )
    _declare_output_dims(path, model.output_shape(1)[1:], "batch")
    return path


def graph_dims(path: Path) -> tuple[tuple[int | str, ...], tuple[int | str, ...]]:
    """Read the declared input and output dimensions of an exported graph.

    The *declared* dims, not the dims of some run's output: a dynamic axis appears here as
    its symbolic name and a fixed axis as its extent, which is the only way to tell the two
    apart without exporting twice at different batch sizes.

    Args:
        path: The ``.onnx`` file.

    Returns:
        ``(input_dims, output_dims)``, each element an ``int`` for a fixed axis or the
        symbol name for a dynamic one -- so a correctly exported graph reads
        ``(("batch", 200, 6), ("batch", 150, 6))``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the graph does not have exactly one input and one output.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no exported graph at {path}")
    import onnx

    graph = onnx.load(str(path)).graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError(
            f"{path} declares {len(graph.input)} inputs and {len(graph.output)} outputs; "
            f"the export contract is one of each"
        )

    def _dims(value: object) -> tuple[int | str, ...]:
        """Render one value-info's shape as fixed extents and symbol names."""
        shape = value.type.tensor_type.shape  # type: ignore[attr-defined]
        return tuple(d.dim_param if d.dim_param else int(d.dim_value) for d in shape.dim)

    return _dims(graph.input[0]), _dims(graph.output[0])


def graph_input_geometry(path: Path) -> tuple[int, int]:
    """Return the fixed ``(L, C_in)`` of an exported graph, samples and channels.

    Lets a consumer that holds only the ``.onnx`` file -- which is the deployment situation,
    and the situation the ORT benchmark children are deliberately put in -- shape its input
    buffer without loading a checkpoint.

    Args:
        path: The ``.onnx`` file.

    Returns:
        ``(lookback, n_input_channels)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the input is not rank 3, or if its sequence or channel axis is
            dynamic. Both are fixed by the export contract, and a graph that lost that is
            not the graph this project benchmarks.
    """
    input_dims, _ = graph_dims(path)
    if len(input_dims) != 3:
        raise ValueError(f"{path}: graph input is rank {len(input_dims)}, expected rank 3")
    lookback, n_channels = input_dims[1], input_dims[2]
    if not isinstance(lookback, int) or not isinstance(n_channels, int):
        raise ValueError(
            f"{path}: sequence and channel axes must be fixed, got {input_dims!r}; a "
            f"dynamic sequence axis blocks kernel selection and buys nothing for a "
            f"fixed-lookback forecaster"
        )
    return lookback, n_channels
