"""Core recurrent nodes (reform step 10).

Provides the ``Network & Layers/Recurrent`` category with **three** nodes:

* ``RNN``   - the vanilla tanh cell, one hidden state
* ``LSTM``  - the long short-term memory cell, hidden + cell states
* ``GRU``   - the gated recurrent unit, reset / update / candidate gates

The cells are written out **explicitly** - the tanh, the three sigmoid gates of
the LSTM, the two gates of the GRU - instead of delegating to ``nn.RNN`` /
``nn.LSTM`` / ``nn.GRU``, exactly like the attention family implements the
softmax by hand: every step of the recurrence is visible in the node's code,
which is the teaching point, and the result is bit-for-bit deterministic on
every backend.

Two ideas the nodes are meant to make visible:

* **Stateless recurrence** - the weights (``weight_ih`` / ``weight_hh`` / the
  two biases, laid out exactly like ``nn.RNNBase``: ``gates * hidden`` rows)
  and the initial states (``h0``, plus ``c0`` for the LSTM) arrive through
  input slots.  Nothing is initialised or remembered inside the node, so one
  node runs any checkpoint and ComfyUI's caching stays meaningful.  Every
  slot except ``x`` is optional: an unconnected weight is a zero weight and an
  unconnected state is a zero state, so a freshly placed node with only ``x``
  wired already runs (producing zero states - wire the weights in for real
  work).
* **One layer, stackable** - the nodes deliberately ship a *single* layer with
  ``batch_first`` input.  A stacked recurrent network is the previous node's
  ``y`` fed into the next node's ``x``, and the per-step hidden state ``hn`` /
  ``cn`` can be round-tripped into the next execution's ``h0`` / ``c0`` - the
  same composition-over-configuration stance as the rest of the core layer
  family (no ``num_layers`` / ``bidirectional`` / ``dropout`` widgets).

The weight layout matches ``torch.nn`` so a checkpoint's
``weight_ih_l0`` / ``weight_hh_l0`` / ``bias_ih_l0`` / ``bias_hh_l0`` tensors
plug straight in (the two biases add, exactly like ``nn``); the gate order is
``[i, f, g, o]`` for the LSTM and ``[r, z, n]`` for the GRU, as in torch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Recurrent"

#: Number of gate rows per hidden unit: RNN 1 (the tanh cell), LSTM 4
#: (input, forget, candidate, output), GRU 3 (reset, update, candidate).
_GATES = {"rnn": 1, "lstm": 4, "gru": 3}


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _recurrent_schema(
    node_id: str,
    display_name: str,
    description: str,
    inputs: list,
    outputs: list,
    search_aliases: list[str],
) -> io.Schema:
    """Build the schema shared by every recurrent node.

    Args:
        node_id: Globally unique, core-safe node id.
        display_name: Short name shown in the node library (``RNN`` / ``LSTM`` / ``GRU``).
        description: Tooltip shown when hovering over the node.
        inputs: The node's inputs, in declaration order (slots and widgets).
        outputs: The node's outputs (``y`` / ``hn`` and, for the LSTM, ``cn``).
        search_aliases: Extra search keywords for the node library.

    Returns:
        The ``io.Schema`` describing the node.
    """
    return io.Schema(
        node_id=node_id,
        display_name=display_name,
        category=CATEGORY,
        description=description,
        search_aliases=search_aliases,
        inputs=list(inputs),
        outputs=list(outputs),
    )


def _hidden_size_widget() -> io.Int.Input:
    return io.Int.Input(
        "hidden_size",
        default=16,
        min=1,
        max=4096,
        step=1,
        tooltip="Hidden units H of the cell; the wired weights must match ((gates * H), in_features) / ((gates * H), H).",
    )


def _x_input() -> io.Tensor.Input:
    return io.Tensor.Input(
        "x",
        tooltip="Input sequence, batch-first (batch, seq_len, in_features); a bare (seq_len, in_features) runs as a batch of one.",
    )


def _weight_input(name: str, shape: str) -> io.Tensor.Input:
    return io.Tensor.Input(
        name,
        optional=True,
        tooltip=f"Optional weight of shape {shape}, wired in as a tensor (the nn.RNNBase layout); unconnected means zeros.",
    )


def _bias_input(name: str, gates: int) -> io.Tensor.Input:
    return io.Tensor.Input(
        name,
        optional=True,
        tooltip=f"Optional bias of shape ({gates} * hidden,), wired in as a tensor; the two biases add, as in nn.RNNBase. Unconnected means zeros.",
    )


def _state_input(name: str, kind: str) -> io.Tensor.Input:
    return io.Tensor.Input(
        name,
        optional=True,
        tooltip=f"Optional initial {kind} of shape (batch, hidden) or (1, batch, hidden); unconnected means zeros.",
    )


def _prepare_state(
    state: torch.Tensor | None,
    batch: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Normalise an initial state to ``(batch, hidden)``, zeros when unconnected.

    Args:
        state: The wired tensor, or ``None``.  Any layout holding exactly
            ``batch * hidden`` values is accepted and reshaped, so both the
            torch ``(1, batch, hidden)`` and the node's own ``(batch, hidden)``
            output round-trip.
        batch: Batch size ``N``.
        hidden: Hidden size ``H``.
        dtype: Common dtype of the run.
        device: Device of the run.
        name: Slot name used in the error message.

    Returns:
        The ``(batch, hidden)`` initial state.

    Raises:
        ValueError: When the wired tensor does not hold ``batch * hidden`` values.
    """
    if state is None:
        return torch.zeros(batch, hidden, dtype=dtype, device=device)
    if state.numel() != batch * hidden:
        raise ValueError(
            f"{name} must hold batch * hidden = {batch} * {hidden} = {batch * hidden} "
            f"value(s), e.g. shape ({batch}, {hidden}) or (1, {batch}, {hidden}); "
            f"got shape {tuple(state.shape)}."
        )
    return state.detach().to(device=device, dtype=dtype).reshape(batch, hidden)


def _recurrent_forward(
    x: torch.Tensor,
    weight_ih: torch.Tensor | None,
    weight_hh: torch.Tensor | None,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    h0: torch.Tensor | None,
    c0: torch.Tensor | None,
    hidden_size: int,
    kind: str,
    node_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run one explicit single-layer recurrent cell over a batch-first sequence.

    The same function backs all three nodes; ``kind`` picks the cell (``rnn`` /
    ``lstm`` / ``gru``) and with it the gate count and the update rule.  The
    input-side and hidden-side transforms are computed separately with
    ``F.linear`` (each carrying its own bias), because the GRU's candidate gate
    needs the reset gate to scale **only** the hidden-side contribution - for
    the RNN and LSTM the two sides simply add, which is the ``nn.RNNBase``
    convention.

    Args:
        x: ``(batch, seq_len, in_features)`` (already batched; leading dims
            beyond the batch are flattened by the caller).
        weight_ih: ``((gates * H), in_features)`` or ``None`` (zeros).
        weight_hh: ``((gates * H), H)`` or ``None`` (zeros).
        bias_ih / bias_hh: ``(gates * H,)`` or ``None`` (zeros).
        h0: ``(batch, H)`` or ``None`` (zeros).
        c0: LSTM only: ``(batch, H)`` or ``None`` (zeros).
        hidden_size: ``H``, from the widget.
        kind: ``"rnn"`` / ``"lstm"`` / ``"gru"``.
        node_name: Display name of the node, used in error messages.

    Returns:
        ``(y, hn, cn)`` - the per-step outputs ``(batch, seq_len, H)``, the
        final hidden state ``(batch, H)`` and (LSTM only) the final cell state
        ``(batch, H)``; ``cn`` is ``None`` for the RNN / GRU.

    Raises:
        ValueError: When a wired weight or state does not match the widget's
            ``hidden_size`` and the input's ``in_features``.
    """
    gates = _GATES[kind]
    batch, steps, in_features = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    hidden = int(hidden_size)

    # 1. dtype / device promotion - the BasicLinear contract: fp16 activations
    #    with fp32 weights run in fp32 instead of raising a mismatch error.
    if not x.is_floating_point():
        x = x.to(dtype=torch.float32)
    parts = [t for t in (weight_ih, weight_hh, bias_ih, bias_hh, h0, c0) if t is not None]
    common = x.dtype
    for t in parts:
        if not t.is_floating_point():
            t_float = t.to(dtype=torch.float32)
        else:
            t_float = t
        common = torch.promote_types(common, t_float.dtype)
    if x.dtype != common:
        x = x.to(dtype=common)
    device = x.device
    if weight_ih is not None:
        weight_ih = weight_ih.to(device=device, dtype=common)
    if weight_hh is not None:
        weight_hh = weight_hh.to(device=device, dtype=common)
    if bias_ih is not None:
        bias_ih = bias_ih.to(device=device, dtype=common)
    if bias_hh is not None:
        bias_hh = bias_hh.to(device=device, dtype=common)

    # 2. weights: shape checks, then unconnected means zeros.
    rows = gates * hidden
    if weight_ih is not None:
        if weight_ih.dim() != 2 or tuple(weight_ih.shape) != (rows, in_features):
            raise ValueError(
                f"{node_name}: weight_ih must be 2-D of shape ({rows}, {in_features}) "
                f"(gates={gates}, hidden={hidden}, in_features={in_features}); "
                f"got shape {tuple(weight_ih.shape)}."
            )
    else:
        weight_ih = torch.zeros(rows, in_features, dtype=common, device=device)
    if weight_hh is not None:
        if weight_hh.dim() != 2 or tuple(weight_hh.shape) != (rows, hidden):
            raise ValueError(
                f"{node_name}: weight_hh must be 2-D of shape ({rows}, {hidden}); "
                f"got shape {tuple(weight_hh.shape)}."
            )
    else:
        weight_hh = torch.zeros(rows, hidden, dtype=common, device=device)
    for name, bias in (("bias_ih", bias_ih), ("bias_hh", bias_hh)):
        if bias is not None and bias.numel() != rows:
            raise ValueError(
                f"{node_name}: {name} must hold {rows} value(s) ({gates} * hidden); "
                f"got {tuple(bias.shape)} with {bias.numel()}."
            )
    if bias_ih is None:
        bias_ih = torch.zeros(rows, dtype=common, device=device)
    if bias_hh is None:
        bias_hh = torch.zeros(rows, dtype=common, device=device)

    # 3. initial states.
    h = _prepare_state(h0, batch, hidden, common, device, "h0")
    c = _prepare_state(c0, batch, hidden, common, device, "c0") if kind == "lstm" else None

    # 4. the explicit recurrence - one loop, one cell update per step.
    outputs: list[torch.Tensor] = []
    for t in range(steps):
        gi = F.linear(x[:, t], weight_ih, bias_ih)  # input side, bias_ih included
        gh = F.linear(h, weight_hh, bias_hh)        # hidden side, bias_hh included
        if kind == "rnn":
            h = torch.tanh(gi + gh)
        elif kind == "lstm":
            i, f, g, o = (gi + gh).chunk(4, dim=1)  # [i, f, g, o] as in torch
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            c = f * c + i * torch.tanh(g)
            h = o * torch.tanh(c)
        else:  # gru
            r_g, z_g, n_g = gi.chunk(3, dim=1)
            r_h, z_h, n_h = gh.chunk(3, dim=1)
            r = torch.sigmoid(r_g + r_h)            # reset gate
            z = torch.sigmoid(z_g + z_h)            # update gate
            n = torch.tanh(n_g + r * n_h)           # candidate: r scales the hidden side only
            h = (1.0 - z) * n + z * h
        outputs.append(h)
    y = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
    return y, h, c


def _run_node(
    x: torch.Tensor,
    weight_ih: torch.Tensor | None,
    weight_hh: torch.Tensor | None,
    bias_ih: torch.Tensor | None,
    bias_hh: torch.Tensor | None,
    h0: torch.Tensor | None,
    c0: torch.Tensor | None,
    hidden_size: int,
    kind: str,
    node_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Execute one recurrent node: flatten leading dims, run, restore, detach.

    ``x`` may carry arbitrary leading batch dimensions ``(…, N, seq_len,
    in_features)``; they are flattened into one batch for the loop and restored
    on ``y`` afterwards, so the node composes with the rest of the family.

    Returns:
        ``(y, hn, cn)`` with ``y`` shaped like ``x`` but ``H`` wide, and the
        detached final states ``(N, H)``.
    """
    if x.dim() < 2:
        raise ValueError(
            f"{node_name}: x must have at least 2 dimensions (batch, seq_len, in_features) "
            f"or (seq_len, in_features); got shape {tuple(x.shape)}."
        )
    hidden = max(1, int(hidden_size))
    if hidden != int(hidden_size):
        _warn(f"{node_name}: hidden_size={hidden_size} is not a positive integer; using {hidden}.")

    leading = x.shape[:-2]
    batched = x.reshape(-1, x.shape[-2], x.shape[-1])
    y, hn, cn = _recurrent_forward(
        batched, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0, hidden, kind, node_name
    )
    y = y.reshape(*leading, y.shape[-2], y.shape[-1])
    return y.detach(), hn.detach(), None if cn is None else cn.detach()


class RecurrentRNN(io.ComfyNode):
    """The vanilla recurrent cell: one tanh update per time step.

    What: ``h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`` written out
          explicitly - the 1986-era recurrence every later gated cell builds
          on. Single layer, batch-first, weights and the initial state wired in
          through slots: the node is stateless, so one node runs any checkpoint
          and stacking is done by feeding this node's ``y`` into the next
          node's ``x``. An unconnected weight or ``h0`` means zeros, so a
          freshly placed node with only ``x`` wired already runs.
    In:   x (TENSOR) - ``(batch, seq_len, in_features)``; a bare
          ``(seq_len, in_features)`` runs as a batch of one, and extra leading
          dimensions are flattened into the batch and restored on ``y``.
          weight_ih (TENSOR, optional) - ``(hidden, in_features)``.
          weight_hh (TENSOR, optional) - ``(hidden, hidden)``.
          bias_ih, bias_hh (TENSOR, optional) - ``(hidden,)`` each; they add,
          exactly like ``nn.RNN``.
          h0 (TENSOR, optional) - ``(batch, hidden)`` or
          ``(1, batch, hidden)``; unconnected means zeros.
          hidden_size (INT) - hidden units ``H`` (default 16).
    Out:  y (TENSOR) - ``(batch, seq_len, hidden)``, the per-step hidden state.
          hn (TENSOR) - ``(batch, hidden)``, the state after the last step;
          feed it into the next execution's ``h0`` to continue a sequence.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _recurrent_schema(
            "RecurrentRNN",
            "RNN",
            "The vanilla tanh recurrent cell, single layer and batch-first: weights, biases and the initial state are wired in as tensors.",
            inputs=[
                _x_input(),
                _weight_input("weight_ih", "(hidden, in_features)"),
                _weight_input("weight_hh", "(hidden, hidden)"),
                _bias_input("bias_ih", 1),
                _bias_input("bias_hh", 1),
                _state_input("h0", "hidden state"),
                _hidden_size_widget(),
            ],
            outputs=[
                io.Tensor.Output(display_name="y"),
                io.Tensor.Output(display_name="hn"),
            ],
            search_aliases=["rnn", "recurrent", "vanilla rnn", "tanh cell", "sequence", "elman"],
        )

    @classmethod
    def execute(
        cls,
        x: torch.Tensor,
        weight_ih: torch.Tensor | None = None,
        weight_hh: torch.Tensor | None = None,
        bias_ih: torch.Tensor | None = None,
        bias_hh: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        hidden_size: int = 16,
    ) -> io.NodeOutput:
        y, hn, _ = _run_node(x, weight_ih, weight_hh, bias_ih, bias_hh, h0, None, hidden_size, "rnn", "RNN")
        return io.NodeOutput(y, hn)


class RecurrentLSTM(io.ComfyNode):
    """The LSTM cell: input, forget, candidate and output gates, two states.

    What: the long short-term memory update, written out explicitly per step -
          ``i, f, g, o = gates(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`` in
          torch's ``[i, f, g, o]`` row order, ``c_t = f * c_{t-1} + i * g``
          and ``h_t = o * tanh(c_t)``. The forget gate is what lets a
          gradient survive across hundreds of steps; the cell state ``c``
          travels on its own slot pair (``c0`` in, ``cn`` out) beside the
          hidden state. Single layer, batch-first, weights and states wired
          in; unconnected slots mean zeros.
    In:   x (TENSOR) - ``(batch, seq_len, in_features)``; a bare
          ``(seq_len, in_features)`` runs as a batch of one, and extra leading
          dimensions are flattened into the batch and restored on ``y``.
          weight_ih (TENSOR, optional) - ``(4 * hidden, in_features)``.
          weight_hh (TENSOR, optional) - ``(4 * hidden, hidden)``.
          bias_ih, bias_hh (TENSOR, optional) - ``(4 * hidden,)`` each; they
          add, exactly like ``nn.LSTM``.
          h0, c0 (TENSOR, optional) - initial hidden / cell state,
          ``(batch, hidden)`` or ``(1, batch, hidden)`` each; unconnected
          means zeros.
          hidden_size (INT) - hidden units ``H`` (default 16).
    Out:  y (TENSOR) - ``(batch, seq_len, hidden)``, ``h_t`` per step.
          hn (TENSOR) - ``(batch, hidden)``, the final hidden state.
          cn (TENSOR) - ``(batch, hidden)``, the final cell state; feed
          ``hn`` / ``cn`` into the next execution's ``h0`` / ``c0`` to
          continue a sequence.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _recurrent_schema(
            "RecurrentLSTM",
            "LSTM",
            "The LSTM cell with its input / forget / candidate / output gates, single layer and batch-first; weights and both initial states are wired in as tensors.",
            inputs=[
                _x_input(),
                _weight_input("weight_ih", "(4 * hidden, in_features)"),
                _weight_input("weight_hh", "(4 * hidden, hidden)"),
                _bias_input("bias_ih", 4),
                _bias_input("bias_hh", 4),
                _state_input("h0", "hidden state"),
                _state_input("c0", "cell state"),
                _hidden_size_widget(),
            ],
            outputs=[
                io.Tensor.Output(display_name="y"),
                io.Tensor.Output(display_name="hn"),
                io.Tensor.Output(display_name="cn"),
            ],
            search_aliases=["lstm", "long short term memory", "recurrent", "gates", "cell state", "sequence"],
        )

    @classmethod
    def execute(
        cls,
        x: torch.Tensor,
        weight_ih: torch.Tensor | None = None,
        weight_hh: torch.Tensor | None = None,
        bias_ih: torch.Tensor | None = None,
        bias_hh: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        c0: torch.Tensor | None = None,
        hidden_size: int = 16,
    ) -> io.NodeOutput:
        y, hn, cn = _run_node(x, weight_ih, weight_hh, bias_ih, bias_hh, h0, c0, hidden_size, "lstm", "LSTM")
        return io.NodeOutput(y, hn, cn)


class RecurrentGRU(io.ComfyNode):
    """The GRU cell: reset and update gates, one state, no cell memory.

    What: the gated recurrent unit, written out explicitly per step in torch's
          ``[r, z, n]`` row order - ``r = sigmoid(...)`` decides how much of
          the previous state the **candidate** ``n = tanh(W_in x + b_in +
          r * (W_hn h + b_hn))`` may look at (the reset gate scales the hidden
          side only, which is why the node computes the input and hidden sides
          separately), and ``z`` interpolates between the candidate and the
          old state: ``h_t = (1 - z) * n + z * h_{t-1}``. The LSTM's separate
          cell state is gone - one state, two gates, a third cheaper cell.
          Single layer, batch-first, weights and the initial state wired in;
          unconnected slots mean zeros.
    In:   x (TENSOR) - ``(batch, seq_len, in_features)``; a bare
          ``(seq_len, in_features)`` runs as a batch of one, and extra leading
          dimensions are flattened into the batch and restored on ``y``.
          weight_ih (TENSOR, optional) - ``(3 * hidden, in_features)``.
          weight_hh (TENSOR, optional) - ``(3 * hidden, hidden)``.
          bias_ih, bias_hh (TENSOR, optional) - ``(3 * hidden,)`` each; they
          add, exactly like ``nn.GRU``.
          h0 (TENSOR, optional) - ``(batch, hidden)`` or
          ``(1, batch, hidden)``; unconnected means zeros.
          hidden_size (INT) - hidden units ``H`` (default 16).
    Out:  y (TENSOR) - ``(batch, seq_len, hidden)``, the per-step hidden state.
          hn (TENSOR) - ``(batch, hidden)``, the state after the last step;
          feed it into the next execution's ``h0`` to continue a sequence.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _recurrent_schema(
            "RecurrentGRU",
            "GRU",
            "The gated recurrent unit with its reset / update gates, single layer and batch-first; weights and the initial state are wired in as tensors.",
            inputs=[
                _x_input(),
                _weight_input("weight_ih", "(3 * hidden, in_features)"),
                _weight_input("weight_hh", "(3 * hidden, hidden)"),
                _bias_input("bias_ih", 3),
                _bias_input("bias_hh", 3),
                _state_input("h0", "hidden state"),
                _hidden_size_widget(),
            ],
            outputs=[
                io.Tensor.Output(display_name="y"),
                io.Tensor.Output(display_name="hn"),
            ],
            search_aliases=["gru", "gated recurrent unit", "recurrent", "reset gate", "update gate", "sequence"],
        )

    @classmethod
    def execute(
        cls,
        x: torch.Tensor,
        weight_ih: torch.Tensor | None = None,
        weight_hh: torch.Tensor | None = None,
        bias_ih: torch.Tensor | None = None,
        bias_hh: torch.Tensor | None = None,
        h0: torch.Tensor | None = None,
        hidden_size: int = 16,
    ) -> io.NodeOutput:
        y, hn, _ = _run_node(x, weight_ih, weight_hh, bias_ih, bias_hh, h0, None, hidden_size, "gru", "GRU")
        return io.NodeOutput(y, hn)


#: Every node this module registers, in node-library order.
RECURRENT_NODES: list[type[io.ComfyNode]] = [
    RecurrentRNN,
    RecurrentLSTM,
    RecurrentGRU,
]


class RecurrentExtension(ComfyExtension):
    """Registers the core recurrent node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(RECURRENT_NODES)


async def comfy_entrypoint() -> RecurrentExtension:
    return RecurrentExtension()
