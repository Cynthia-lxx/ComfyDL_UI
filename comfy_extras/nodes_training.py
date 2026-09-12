"""Core training nodes (reform step 6): learnable parameters, optimizers, a trainer.

ComfyUI runs a prompt inside ``torch.inference_mode()`` (``execution.py``), so an
autograd graph cannot cross a node boundary: no node can differentiate a tensor
produced by another node. Every gradient based workflow therefore has to keep
forward, backward and ``optimizer.step()`` inside *one* node - which is exactly
what upstream's ``TrainLoraNode`` does. This module provides the pieces of that
closure as ordinary, composable nodes:

* two graph value types, declared in ``comfy_api``:

  - ``PARAMS``    - a named trainable parameter set: ``{name: nn.Parameter}``.
    Names follow the module / ``safetensors`` convention, so an entry can be
    pulled back out as a plain tensor and fed to the stateless ``Basic`` layer
    nodes (``layer0.weight`` into ``Linear.weight``), saved to disk, or encoded
    into a string widget. This is what makes "train, then infer" a single graph.
  - ``OPTIMIZER`` - the hyper-parameters of a ``torch.optim`` optimizer. It is a
    *configuration* object, not a live optimizer: the trainer builds the real
    optimizer inside its own call, which keeps the value free of device state
    and safe for ComfyUI to cache.

* ``Learnable Parameters`` / ``Merge Parameters`` / ``Parameters to Tensor`` -
  create a parameter set from a tensor or from a shape, combine two sets, and
  take one entry back out as a tensor.
* ``Optimizer`` - publishes the optimizer settings (hyper-parameters live on
  widgets, the value travels on a link, so flipping a widget invalidates the
  graph exactly like ``TrainingMode`` does).
* ``Training Loop`` - the trainer: builds a small MLP from the ``hidden`` widget,
  then runs ``steps`` iterations of forward / backward / ``optimizer.step()``
  inside a ``torch.inference_mode(False)`` block and returns the trained
  parameters, the last loss, the per-step loss history and the predictions.
* ``Save Parameters`` / ``Load Parameters`` - the file half of the persistence
  story, plus ``Parameters to Text`` / ``Text to Parameters`` for the widget
  half, which survives inside a saved workflow without touching the disk.

Everything is deterministic for a given ``seed`` and never touches the process
RNG outside its own seeded block, so caching a node's output stays correct.
"""

import dataclasses
import os
import re

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing_extensions import override

import comfy.utils
import folder_paths
from comfy import training_protocol as tp
from comfy_api.latest import ComfyExtension, io

CATEGORY = "Network & Layers/Training"

#: How a freshly dropped ``Learnable Parameters`` node initialises its tensor.
INIT_OPTIONS: tuple[str, ...] = ("normal", "zeros", "ones", "xavier_uniform", "kaiming_uniform")

#: Loss functions the trainer can minimize. ``cross_entropy`` expects class
#: targets (indices or one-hot), the two others regression targets.
LOSS_OPTIONS: tuple[str, ...] = ("mse", "l1", "cross_entropy")

#: Default of the ``shape`` widget, and the value a typo falls back to.
DEFAULT_SHAPE = "2,3"

#: Default of the ``hidden`` widget, and the value a typo falls back to.
DEFAULT_HIDDEN = "8"

#: Default output folder prefix of ``Save Parameters``; ``Load Parameters``
#: defaults to the first file that a save with this prefix produces.
DEFAULT_FILE_PREFIX = "comfydl/parameters"

#: Above this many iterations the trainer warns: the loop is Python, and a user
#: asking for a million steps has almost certainly typed one zero too many.
STEPS_WARN_THRESHOLD = 20000

#: Longest run of names printed in one log line before it is cut short.
NAME_PREVIEW = 8


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _parse_sizes(text: str) -> tuple[int, ...] | None:
    """Parse ``"8"`` or ``"16,8"`` into a tuple of positive ints.

    Accepts comma, semicolon, whitespace and full-width separators so that a
    value typed on a Chinese IME (``"16，8"``) still works.

    Args:
        text: The raw widget value.

    Returns:
        The parsed widths with every non-positive entry dropped, or ``None``
        when the string is empty or holds a non-integer entry; callers fall back
        to their documented default so that a typo cannot break a workflow.
    """
    if text is None:
        return None
    normalized = str(text).replace("，", ",").replace("；", ",").replace(";", ",")
    parts = [part for part in re.split(r"[,\s]+", normalized) if part]
    if not parts:
        return None
    sizes: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            return None
        if value > 0:
            sizes.append(value)
    return tuple(sizes)


def _text(value, fallback: str) -> str:
    """Return a stripped widget string, or ``fallback`` when it is blank."""
    text = "" if value is None else str(value).strip()
    return text or fallback


def _init_tensor(sizes: tuple[int, ...], mode: str) -> torch.Tensor:
    """Create an untrained tensor of ``sizes`` using the selected initialiser.

    Args:
        sizes: Shape of the parameter, every entry positive.
        mode: One of :data:`INIT_OPTIONS`. ``xavier_uniform`` / ``kaiming_uniform``
            need at least two dimensions (they describe fan-in / fan-out); on a
            1-D shape they fall back to ``normal`` with a warning.

    Returns:
        A fresh float32 tensor, already detached and contiguous.
    """
    if mode == "zeros":
        return torch.zeros(sizes, dtype=torch.float32)
    if mode == "ones":
        return torch.ones(sizes, dtype=torch.float32)
    tensor = torch.empty(sizes, dtype=torch.float32)
    if mode in ("xavier_uniform", "kaiming_uniform") and len(sizes) >= 2:
        if mode == "xavier_uniform":
            return nn.init.xavier_uniform_(tensor)
        return nn.init.kaiming_uniform_(tensor)
    if mode in ("xavier_uniform", "kaiming_uniform"):
        _warn(
            f"init '{mode}' needs at least two dimensions, but the shape is "
            f"{tuple(sizes)}; using 'normal' instead."
        )
    elif mode != "normal":
        _warn(f"init '{mode}' is unknown; using 'normal' instead.")
    return torch.randn(sizes, dtype=torch.float32)


def _loss_mode(value: str) -> str:
    """Return a valid loss name, falling back to ``mse`` with a warning."""
    text = "" if value is None else str(value).strip().lower()
    if text in LOSS_OPTIONS:
        return text
    _warn(f"loss '{value}' is unknown; using 'mse' (one of {', '.join(LOSS_OPTIONS)}).")
    return "mse"


def _hidden_sizes(value: str) -> tuple[int, ...]:
    """Read the ``hidden`` widget into the widths of the hidden layers.

    Args:
        value: The raw widget value; empty means "no hidden layer", i.e. a single
            dense layer from the features straight to the outputs.

    Returns:
        The hidden widths, without the input / output layer. Text that cannot be
        read falls back to the default ``8``, and non-positive entries are
        dropped, both with a warning, so a typo cannot break a workflow.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    sizes = _parse_sizes(text)
    if sizes is None:
        _warn(
            f"hidden '{value}' could not be read as layer widths; using "
            f"'{DEFAULT_HIDDEN}' instead."
        )
        sizes = _parse_sizes(DEFAULT_HIDDEN) or (8,)
    return tuple(sizes)


@dataclasses.dataclass(frozen=True)
class _TrainingData:
    """The flattened ``(x, y)`` pair a trainer consumes.

    What: turns the two tensors that arrive on the ``x`` / ``y`` slots into the
          ``(samples, features)`` / ``(samples, targets)`` form the loop needs,
          and rejects the combinations that cannot be trained at all. The rules
          are the ones the layer nodes already use - the *last* dimension is the
          feature dimension - so a trainer can sit behind exactly the same nodes
          as a ``Linear`` layer.
    In:   the raw ``x`` / ``y`` tensors plus the selected loss name.
    Out: ``inputs`` (float32, ``(N, features)``), ``targets`` (float32
         ``(N, outputs)`` for a regression loss, int64 ``(N,)`` class indices for
         ``cross_entropy``), the feature / output counts, the sample count and
         the loss name.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    features: int
    outputs: int
    samples: int
    loss: str

    @property
    def classification(self) -> bool:
        """True when the targets are class indices rather than values."""
        return self.loss == "cross_entropy"

    @classmethod
    def from_inputs(cls, x: torch.Tensor, y: torch.Tensor, loss: str) -> "_TrainingData":
        """Flatten and validate ``x`` / ``y``; see the class docstring.

        Raises:
            TypeError: when a slot did not receive a tensor.
            ValueError: when the pair cannot be trained - empty tensors, a
                different number of samples in ``x`` and ``y``, or class targets
                that are negative or not integral.
        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(
                "[Network & Layers] Training Loop needs a tensor on both the 'x' "
                f"and the 'y' slot; got {type(x).__name__} and {type(y).__name__}."
            )
        mode = _loss_mode(loss)
        inputs = _flatten_features(x)
        samples, features = (int(inputs.shape[0]), int(inputs.shape[1]))
        if samples == 0 or features == 0:
            raise ValueError(
                f"[Network & Layers] Training Loop got an empty input of shape "
                f"{tuple(x.shape)}; 'x' must hold at least one sample and one feature."
            )
        if mode == "cross_entropy":
            targets, outputs = _class_targets(y, samples)
        else:
            targets = _flatten_features(y).to(dtype=torch.float32)
            if int(targets.shape[0]) != samples:
                raise ValueError(
                    f"[Network & Layers] Training Loop needs as many targets as "
                    f"inputs, but 'x' holds {samples} sample(s) and 'y' holds "
                    f"{int(targets.shape[0])}."
                )
            outputs = int(targets.shape[1])
        inputs = inputs.to(dtype=torch.float32)
        return cls(
            inputs=inputs,
            targets=targets.to(device=inputs.device),
            features=features,
            outputs=outputs,
            samples=samples,
            loss=mode,
        )


def _flatten_features(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten every dimension but the last one: ``(..., features)`` -> ``(N, features)``.

    A 1-D tensor is read as a single feature column, matching the way
    ``torch.nn.Linear`` treats a bare vector.
    """
    if tensor.dim() == 0:
        raise ValueError(
            "[Network & Layers] Training Loop got a scalar; a sample needs at least "
            "one feature dimension."
        )
    if tensor.dim() == 1:
        return tensor.reshape(-1, 1)
    return tensor.reshape(-1, tensor.shape[-1])


def _class_targets(y: torch.Tensor, samples: int) -> tuple[torch.Tensor, int]:
    """Read ``y`` as classification targets and return ``(indices, classes)``.

    Accepts the two spellings PyTorch's ``cross_entropy`` understands: class
    indices, or a 2-D one-hot / probability tensor whose last dimension is the
    class count. A scalar target is rejected - it cannot be matched with ``x``.
    """
    if y.dim() >= 2 and y.shape[-1] > 1 and y.is_floating_point():
        probabilities = y.reshape(-1, y.shape[-1])
        if int(probabilities.shape[0]) != samples:
            raise ValueError(
                f"[Network & Layers] Training Loop got {samples} input sample(s) but "
                f"{int(probabilities.shape[0])} one-hot target(s)."
            )
        return probabilities.argmax(dim=-1).to(dtype=torch.long), int(probabilities.shape[1])
    indices = y.reshape(-1)
    if int(indices.shape[0]) != samples:
        raise ValueError(
            f"[Network & Layers] Training Loop got {samples} input sample(s) but "
            f"{int(indices.shape[0])} class target(s)."
        )
    if indices.is_floating_point():
        rounded = indices.round()
        if not bool(torch.equal(indices.detach().cpu(), rounded.detach().cpu())):
            raise ValueError(
                "[Network & Layers] with loss='cross_entropy' the targets on 'y' must "
                "be class indices (integers), or a 2-D one-hot / probability tensor; "
                "float targets are only accepted when they are integral."
            )
        indices = rounded
    indices = indices.to(dtype=torch.long)
    smallest = int(indices.min())
    if smallest < 0:
        raise ValueError(
            f"[Network & Layers] with loss='cross_entropy' class indices must be >= 0, "
            f"but 'y' holds {smallest}."
        )
    return indices, int(indices.max()) + 1


def _loss_value(mode: str, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Apply the selected loss function to a batch."""
    if mode == "l1":
        return F.l1_loss(prediction, targets)
    if mode == "cross_entropy":
        return F.cross_entropy(prediction, targets)
    return F.mse_loss(prediction, targets)


def _summarize_warm_start(
    loaded: list[str], missing: list[str], skipped: list[str]
) -> str:
    """One-line report of a warm start, so nothing is dropped silently."""
    parts = [f"warm start: {len(loaded)} parameter(s) loaded"]
    if missing:
        parts.append(
            f"left at their fresh values: "
            f"{tp.parameter_names({name: None for name in missing}, NAME_PREVIEW)}"
        )
    if skipped:
        shown = "; ".join(skipped[:NAME_PREVIEW])
        if len(skipped) > NAME_PREVIEW:
            shown += f"; ... (+{len(skipped) - NAME_PREVIEW} more)"
        parts.append(f"ignored: {shown}")
    return " | ".join(parts)


def _resolve_parameter_path(path_text: str) -> str:
    """Turn the ``path`` widget into an absolute file path.

    A relative path is resolved against ComfyUI's *output* folder, which is where
    ``Save Parameters`` writes; an absolute path is used as it is.
    """
    text = "" if path_text is None else str(path_text).strip()
    if not text:
        raise ValueError(
            "[Network & Layers] the 'path' widget is empty; give a .safetensors "
            "file, either absolute or relative to the output folder."
        )
    expanded = os.path.expanduser(text)
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(folder_paths.get_output_directory(), expanded))


def _save_parameter_file(params, filename_prefix: str) -> str:
    """Write a parameter set to a uniquely named ``.safetensors`` in ``output/``.

    Args:
        params: The payload to write (any ``PARAMS`` value).
        filename_prefix: Widget value; may contain a subfolder and ``%date%``
            style placeholders, resolved by ``folder_paths.get_save_image_path``.

    Returns:
        The absolute path that was written.
    """
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, _subfolder, _prefix = (
        folder_paths.get_save_image_path(filename_prefix, output_dir)
    )
    os.makedirs(full_output_folder, exist_ok=True)
    output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
    comfy.utils.save_torch_file(tp.parameters_to_tensors(params), output_path)
    return output_path


def _parameters_from_state(state) -> tuple[dict[str, nn.Parameter], list[str], list[str]]:
    """Wrap a loaded ``safetensors`` state dict into a trainable parameter set.

    Returns:
        ``(params, dropped, converted)`` - the parameters, the keys that were not
        floating point tensors (they can never be trained, so they are reported
        instead of kept), and the keys whose dtype had to be promoted to float32.
    """
    params: dict[str, nn.Parameter] = {}
    dropped: list[str] = []
    converted: list[str] = []
    with torch.inference_mode(False):
        for key, value in state.items():
            name = str(key)
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                dropped.append(name)
                continue
            detached = value.detach().clone()
            if detached.dtype != torch.float32:
                detached = detached.float()
                converted.append(name)
            params[name] = nn.Parameter(detached, requires_grad=True)
    return params, dropped, converted


#: Default of the text widget of ``Text to Parameters``: a valid, decodable blob
#: holding a 2x3 ``weight``, so the node produces a parameter set without any
#: editing at all. Built once at import time (a few dozen bytes).
DEFAULT_PARAMS_TEXT = tp.encode_parameters({"weight": torch.zeros(2, 3)})

#: Default of the ``path`` widget of ``Load Parameters``: the first file a save
#: with :data:`DEFAULT_FILE_PREFIX` writes, so a save followed by a load works
#: without anyone typing a file name.
DEFAULT_LOAD_PATH = f"{DEFAULT_FILE_PREFIX}_00001_.safetensors"


class TrainingParameters(io.ComfyNode):
    """Creates a trainable parameter set (``PARAMS``), from a tensor or from scratch.

    What: the entry point of every manual training graph. It turns one tensor into
          a ``nn.Parameter`` wrapped in a ``{name: parameter}`` payload, which is
          what the trainer accepts for warm starting and what ``Parameters to
          Tensor``, ``Save Parameters`` and ``Parameters to Text`` consume.
          Without a wired tensor the value is created here from the ``shape``
          widget, seeded by ``seed``: the same seed always produces the same
          numbers and the process RNG is left untouched, so ComfyUI may cache the
          output.
    In:   tensor (TENSOR, optional) - when connected, this tensor *is* the
          parameter (link wins over the shape widget, exactly like
          ``Training Run Stats``); non floating point tensors are cast to float32
          with a warning, because only floating point tensors can carry a
          gradient.
          name (STRING) - the key of the parameter, e.g. ``"layer0.weight"``;
          the name is what every other node in this family addresses it by.
          shape (STRING) - dimensions to create, e.g. ``"2,3"``; ignored while a
          tensor is wired in. Unreadable text falls back to ``"2,3"`` with a
          warning.
          init (COMBO) - ``normal`` (N(0, 1)), ``zeros``, ``ones``,
          ``xavier_uniform`` or ``kaiming_uniform``; ignored while a tensor is
          wired in.
          seed (INT) - seed of the initialiser, for a reproducible value.
    Out:  params (PARAMS) - a one entry parameter set holding ``name``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingParameters",
            display_name="Learnable Parameters",
            category=CATEGORY,
            description="Creates a named trainable parameter: from a wired tensor, or from the shape/init widgets.",
            search_aliases=[
                "parameter", "learnable", "weights", "variable", "requires_grad",
                "trainable", "init",
            ],
            inputs=[
                io.Tensor.Input(
                    "tensor",
                    optional=True,
                    tooltip="Optional tensor to use as the parameter; overrides shape/init.",
                ),
                io.String.Input(
                    "name",
                    default="weight",
                    placeholder="e.g. layer0.weight",
                    tooltip="Key of the parameter, e.g. layer0.weight; other nodes address it by this name.",
                ),
                io.String.Input(
                    "shape",
                    default=DEFAULT_SHAPE,
                    placeholder="e.g. 2,3",
                    tooltip="Dimensions to create when no tensor is wired in, e.g. 2,3.",
                ),
                io.Combo.Input(
                    "init",
                    options=list(INIT_OPTIONS),
                    default="normal",
                    tooltip="Initialiser used when no tensor is wired in.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    tooltip="Seed of the initialiser; the same seed always yields the same parameter.",
                ),
            ],
            outputs=[io.Params.Output(display_name="params")],
        )

    @classmethod
    def execute(
        cls,
        tensor: torch.Tensor | None = None,
        name: str = "weight",
        shape: str = DEFAULT_SHAPE,
        init: str = "normal",
        seed: int = 0,
    ) -> io.NodeOutput:
        key = _text(name, "weight")
        if tensor is not None:
            value = tensor.detach().clone()
            if not value.is_floating_point():
                _warn(
                    f"parameter '{key}' was given a {value.dtype} tensor; casting to "
                    "float32 because only floating point tensors can be trained."
                )
                value = value.float()
            elif value.dtype != torch.float32:
                value = value.float()
        else:
            sizes = _parse_sizes(shape)
            if sizes is None or not sizes:
                _warn(
                    f"shape '{shape}' could not be read as dimensions; using "
                    f"'{DEFAULT_SHAPE}' instead."
                )
                sizes = _parse_sizes(DEFAULT_SHAPE) or (2, 3)
            mode = _text(init, "normal").lower()
            if mode not in INIT_OPTIONS:
                _warn(f"init '{init}' is unknown; using 'normal' instead.")
                mode = "normal"
            with tp.seeded_rng(seed):
                value = _init_tensor(tuple(sizes), mode)
        # ``nn.Parameter`` must not be built on an inference tensor, which is what
        # every tensor produced inside ComfyUI's own inference_mode block is.
        with torch.inference_mode(False):
            parameter = nn.Parameter(value.detach().clone(), requires_grad=True)
        return io.NodeOutput({key: parameter})


class TrainingParametersMerge(io.ComfyNode):
    """Merges two parameter sets into one.

    What: builds a larger parameter set out of two, e.g. the ``weight`` of one
          ``Learnable Parameters`` node and the ``bias`` of another, or two halves
          of a checkpoint. Keys are kept as they are; a key that appears in both
          inputs is *renamed* (``bias`` -> ``bias_2``) with a warning instead of
          overwriting, so no weight is ever lost silently.
    In:   params_a (PARAMS) - first parameter set; its entries keep their names.
          params_b (PARAMS) - second parameter set, merged after the first one.
    Out:  params (PARAMS) - both sets, in order; chain two nodes to merge three or
          more sets.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingParametersMerge",
            display_name="Merge Parameters",
            category=CATEGORY,
            description="Concatenates two parameter sets; colliding names are renamed with a warning.",
            search_aliases=["concat", "combine", "join", "parameter set", "merge"],
            inputs=[
                io.Params.Input(
                    "params_a",
                    display_name="params a",
                    tooltip="First parameter set; its names are kept as they are.",
                ),
                io.Params.Input(
                    "params_b",
                    display_name="params b",
                    tooltip="Second parameter set; a name already present in 'params a' is renamed.",
                ),
            ],
            outputs=[io.Params.Output(display_name="params")],
        )

    @classmethod
    def execute(cls, params_a, params_b) -> io.NodeOutput:
        merged = dict(tp.as_parameter_dict(params_a))
        for key, value in tp.as_parameter_dict(params_b).items():
            if key in merged:
                renamed = f"{key}_2"
                index = 2
                while renamed in merged:
                    index += 1
                    renamed = f"{key}_{index}"
                _warn(
                    f"parameter '{key}' exists in both inputs; the second one was "
                    f"renamed to '{renamed}'."
                )
                key = renamed
            merged[key] = value
        return io.NodeOutput(merged)


class TrainingParametersExtract(io.ComfyNode):
    """Takes one entry of a parameter set back out, as a plain tensor.

    What: the bridge from training back to inference. A trained parameter set is
          only useful once a single weight matrix can be handed to the stateless
          layer nodes (``Linear`` takes its ``weight`` as a tensor), so this node
          looks an entry up by name and detaches it.
    In:   params (PARAMS) - the parameter set to read.
          name (STRING) - the key to look up, e.g. ``"layer0.weight"``. The
          trainer names its layers ``layer0``, ``layer1``, ... with ``weight`` /
          ``bias`` per layer.
    Out:  output (TENSOR) - the requested tensor, detached (it carries no
          autograd state), on the device it lives on.
    Raises:
        ValueError: when the name is not part of the set; the message lists the
            names that are available.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingParametersExtract",
            display_name="Parameters to Tensor",
            category=CATEGORY,
            description="Pulls one named entry out of a parameter set as a tensor, e.g. layer0.weight for a Linear node.",
            search_aliases=["get parameter", "weight", "bias", "extract", "tensor"],
            inputs=[
                io.Params.Input("params", tooltip="Parameter set to read."),
                io.String.Input(
                    "name",
                    default="weight",
                    placeholder="e.g. layer0.weight",
                    tooltip="Key to look up, e.g. layer0.weight.",
                ),
            ],
            outputs=[io.Tensor.Output(display_name="output")],
        )

    @classmethod
    def execute(cls, params, name: str = "weight") -> io.NodeOutput:
        payload = tp.as_parameter_dict(params)
        key = _text(name, "weight")
        if key not in payload:
            raise ValueError(
                f"[Network & Layers] no parameter named '{key}' in this parameter "
                f"set; available: {tp.parameter_names(payload, NAME_PREVIEW)}."
            )
        return io.NodeOutput(payload[key].detach())


class TrainingOptimizer(io.ComfyNode):
    """Publishes the settings of a ``torch.optim`` optimizer as an ``OPTIMIZER`` value.

    What: the hyper-parameter half of the training loop, kept on widgets while the
          value travels on a link. That split is deliberate: the link is part of
          ComfyUI's cache signature, so changing the learning rate re-runs every
          trainer that consumes it, and one optimizer node can drive several
          trainers at once. The value is a *configuration*, not a live optimizer -
          the trainer builds the real optimizer, so no device state is cached and
          nothing leaks between prompts.
    In:   optimizer (COMBO) - ``AdamW``, ``Adam``, ``SGD`` or ``RMSprop``.
          lr (FLOAT) - learning rate, 0..1 (0.01 by default; AdamW's own default
          of 1e-3 needs more steps than the trainer defaults to).
          momentum (FLOAT) - SGD / RMSprop momentum.
          beta1 / beta2 (FLOAT) - Adam / AdamW decay rates. RMSprop reads
          ``beta2`` as its ``alpha`` (the decay of the running average), so one
          widget covers both.
          eps (FLOAT) - numerical floor of Adam / AdamW / RMSprop.
          weight_decay (FLOAT) - L2 penalty (AdamW applies it decoupled).
          amsgrad (BOOLEAN) - AMSGrad variant of Adam / AdamW.
    Out:  optimizer (OPTIMIZER) - link this into the ``optimizer`` slot of a
          ``Training Loop`` node.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingOptimizer",
            display_name="Optimizer",
            category=CATEGORY,
            description="Publishes the optimizer settings (type, lr, betas, eps, weight decay) for a Training Loop node.",
            search_aliases=[
                "sgd", "adam", "adamw", "rmsprop", "learning rate", "lr",
                "momentum", "weight decay",
            ],
            inputs=[
                io.Combo.Input(
                    "optimizer",
                    options=list(tp.OPTIMIZER_OPTIONS),
                    default="AdamW",
                    tooltip="Optimizer to use. AdamW is the safest default for small networks.",
                ),
                io.Float.Input(
                    "lr",
                    default=tp.DEFAULT_LR,
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    tooltip="Learning rate (0.01 by default).",
                ),
                io.Float.Input(
                    "momentum",
                    default=0.9,
                    min=0.0,
                    max=0.999,
                    step=0.001,
                    tooltip="Momentum of SGD and RMSprop; ignored by Adam and AdamW.",
                ),
                io.Float.Input(
                    "beta1",
                    default=0.9,
                    min=0.0,
                    max=0.999,
                    step=0.001,
                    tooltip="Decay rate of the first moment (Adam / AdamW).",
                ),
                io.Float.Input(
                    "beta2",
                    default=0.999,
                    min=0.0,
                    max=0.9999,
                    step=0.0001,
                    tooltip="Decay rate of the second moment (Adam / AdamW); RMSprop uses it as alpha.",
                ),
                io.Float.Input(
                    "eps",
                    default=1e-8,
                    min=0.0,
                    max=0.001,
                    step=1e-8,
                    tooltip="Numerical floor of the denominator (Adam / AdamW / RMSprop).",
                ),
                io.Float.Input(
                    "weight_decay",
                    default=tp.DEFAULT_WEIGHT_DECAY,
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    tooltip="L2 penalty; AdamW applies it decoupled from the gradient.",
                ),
                io.Boolean.Input(
                    "amsgrad",
                    default=False,
                    tooltip="Use the AMSGrad variant (Adam / AdamW).",
                ),
            ],
            outputs=[io.Optimizer.Output(display_name="optimizer")],
        )

    @classmethod
    def execute(
        cls,
        optimizer: str = "AdamW",
        lr: float = tp.DEFAULT_LR,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = tp.DEFAULT_WEIGHT_DECAY,
        amsgrad: bool = False,
    ) -> io.NodeOutput:
        config = tp.optimizer_config(
            optimizer,
            lr=lr,
            momentum=momentum,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        return io.NodeOutput(config)


class TrainingLoop(io.ComfyNode):
    """Trains a small fully connected network on ``x`` / ``y``, inside the node.

    What: the training closure. ComfyUI executes every node under
          ``torch.inference_mode()`` and caches the result, so a gradient cannot
          survive a node boundary: this node therefore runs forward, backward and
          ``optimizer.step()`` itself, for ``steps`` iterations, inside a
          ``torch.inference_mode(False)`` block (the same technique upstream's
          ``TrainLoraNode`` uses). It builds an MLP from ``hidden`` / ``activation``
          exactly like ``nn.Linear`` stacking does - ``in_features`` from ``x``,
          ``out_features`` from ``y``, activation between the hidden layers only -
          and returns the trained parameters, so the graph can be split anywhere
          between "train" and "use".
          Every output is detached: no autograd graph is left in ComfyUI's cache.
    In:   x (TENSOR) - inputs, ``(N, in_features)`` or any shape whose *last*
          dimension is the feature dimension (a 1-D tensor is read as a single
          feature column). Cast to float32.
          y (TENSOR) - targets. For ``mse`` / ``l1``: ``(N, out_features)``, or a
          1-D tensor for a single output. For ``cross_entropy``: class indices,
          or a 2-D one-hot / probability tensor, whose last dimension is the
          class count.
          optimizer (OPTIMIZER) - settings published by an ``Optimizer`` node.
          params (PARAMS, optional) - warm start: entries whose name *and* shape
          match the freshly built network are loaded, every other entry is
          reported and ignored. Unconnected means a fresh initialisation.
          hidden (STRING) - widths of the hidden layers, e.g. ``"8"`` or
          ``"16,8"``; empty means no hidden layer, i.e. plain linear regression.
          Unreadable text falls back to ``"8"`` with a warning.
          activation (COMBO) - applied between the hidden layers; ``none`` makes
          the whole network linear.
          loss (COMBO) - ``mse``, ``l1`` or ``cross_entropy``.
          steps (INT) - number of optimizer steps (200 by default); each step is
          one forward + backward + update.
          batch_size (INT) - samples per step; ``0`` (default) uses the whole
          dataset every step, the most stable choice for small teaching data.
          seed (INT) - seeds the initialisation and the batch shuffling, so the
          same inputs always produce the same run.
    Out:  params (PARAMS) - the trained parameters, named ``layer0.weight``,
          ``layer0.bias``, ``layer1.weight``, ... in layer order. Feed them to
          ``Parameters to Tensor`` to run inference with the ``Basic`` layer
          nodes, to ``Save Parameters``, or back into another ``Training Loop``
          to continue training.
          loss (TENSOR) - scalar, the last step's loss.
          loss_history (TENSOR) - 1-D, one entry per step; the convergence curve,
          ready for any of the visualisation nodes.
          prediction (TENSOR) - ``model(x)`` of the trained network, detached.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingLoop",
            display_name="Training Loop",
            category=CATEGORY,
            description="Trains a small MLP on x/y inside the node (forward + backward + optimizer.step) and returns the trained parameters, loss history and predictions.",
            search_aliases=[
                "train", "fit", "gradient descent", "backward", "optimizer",
                "regression", "classifier", "mlp", "epoch", "loss",
            ],
            inputs=[
                io.Tensor.Input(
                    "x",
                    tooltip="Inputs, shape (N, in_features); the last dimension is the feature dimension.",
                ),
                io.Tensor.Input(
                    "y",
                    tooltip="Targets: (N, out_features) for mse/l1, or class indices / one-hot for cross_entropy.",
                ),
                io.Optimizer.Input(
                    "optimizer",
                    tooltip="Optimizer settings, linked from an Optimizer node.",
                ),
                io.Params.Input(
                    "params",
                    optional=True,
                    tooltip="Optional starting parameters (warm start); entries with a matching name and shape are loaded.",
                ),
                io.String.Input(
                    "hidden",
                    default=DEFAULT_HIDDEN,
                    placeholder="e.g. 8 or 16,8 (empty = no hidden layer)",
                    tooltip="Widths of the hidden layers, e.g. 8 or 16,8; empty for plain linear regression.",
                ),
                io.Combo.Input(
                    "activation",
                    options=list(tp.ACTIVATION_OPTIONS),
                    default="relu",
                    tooltip="Activation applied between the hidden layers; the output layer stays linear.",
                ),
                io.Combo.Input(
                    "loss",
                    options=list(LOSS_OPTIONS),
                    default="mse",
                    tooltip="Loss: mse / l1 for regression targets, cross_entropy for class targets.",
                ),
                io.Int.Input(
                    "steps",
                    default=200,
                    min=1,
                    max=100000,
                    step=1,
                    tooltip="Number of optimizer steps (one forward + backward + update each).",
                ),
                io.Int.Input(
                    "batch_size",
                    default=0,
                    min=0,
                    max=65536,
                    step=1,
                    tooltip="Samples per step; 0 uses the whole dataset every step.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=4294967295,
                    step=1,
                    tooltip="Seeds the initialisation and the batch shuffling.",
                ),
            ],
            outputs=[
                io.Params.Output(display_name="params"),
                io.Tensor.Output(display_name="loss"),
                io.Tensor.Output(display_name="loss_history"),
                io.Tensor.Output(display_name="prediction"),
            ],
        )

    @classmethod
    def execute(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer,
        params=None,
        hidden: str = DEFAULT_HIDDEN,
        activation: str = "relu",
        loss: str = "mse",
        steps: int = 200,
        batch_size: int = 0,
        seed: int = 0,
    ) -> io.NodeOutput:
        data = _TrainingData.from_inputs(x, y, loss)
        widths = (data.features,) + _hidden_sizes(hidden) + (data.outputs,)

        if not isinstance(optimizer, tp.OptimizerConfig):
            _warn(
                "the 'optimizer' slot did not receive an Optimizer node; using the "
                "default AdamW settings."
            )
            optimizer = tp.optimizer_config()

        iterations = max(1, int(steps))
        if iterations > STEPS_WARN_THRESHOLD:
            _warn(
                f"{iterations} steps were requested; the loop is plain Python and "
                "shares the machine with the rest of the graph."
            )
        per_step = int(batch_size)
        if per_step <= 0 or per_step >= data.samples:
            if per_step > data.samples:
                _warn(
                    f"batch_size {per_step} is larger than the {data.samples} "
                    "sample(s); using the whole dataset every step."
                )
            per_step = 0

        device = data.inputs.device
        shuffler = torch.Generator(device="cpu")
        shuffler.manual_seed(int(seed))
        notes: list[str] = []
        history: list[torch.Tensor] = []

        # Everything below runs outside ComfyUI's inference_mode: an inference
        # tensor cannot take part in a backward pass, and the parameters built
        # here have to stay ordinary (grad carrying) tensors.
        with torch.inference_mode(False):
            model = tp.build_mlp(widths, activation, seed=seed, device=device)
            if params is not None:
                loaded, missing, skipped = tp.load_into_module(
                    model, tp.as_parameter_dict(params)
                )
                notes.append(_summarize_warm_start(loaded, missing, skipped))
            trainer = tp.build_optimizer(optimizer, model.parameters())
            for _ in range(iterations):
                if per_step:
                    order = torch.randperm(data.samples, generator=shuffler)[:per_step].to(device)
                    inputs = data.inputs.index_select(0, order)
                    targets = data.targets.index_select(0, order)
                else:
                    inputs, targets = data.inputs, data.targets
                step_loss = _loss_value(data.loss, model(inputs), targets)
                trainer.zero_grad(set_to_none=True)
                step_loss.backward()
                trainer.step()
                # Detached scalars, stacked into one tensor at the end: the loss
                # history must not keep the graph (or the dataset) alive.
                history.append(step_loss.detach().reshape(()))
            trainer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.grad = None
            curve = torch.stack(history)
            with torch.no_grad():
                predictions = model(data.inputs).detach()

        summary = ", ".join(str(width) for width in widths)
        print(
            f"[Network & Layers] Training Loop: {model.depth} dense layer(s) [{summary}], "
            f"{data.samples} sample(s) x {data.features} feature(s) -> {data.outputs} "
            f"output(s), loss {data.loss}, {iterations} step(s), "
            f"{per_step or data.samples} sample(s) per step, {optimizer.describe()}, "
            f"loss {float(history[0]):.6g} -> {float(curve[-1]):.6g}"
            + (f" | {' | '.join(notes)}" if notes else "")
        )
        return io.NodeOutput(tp.module_parameters(model), curve[-1], curve, predictions)


class TrainingSaveParameters(io.ComfyNode):
    """Writes a parameter set to a ``.safetensors`` file in ComfyUI's output folder.

    What: the file half of the persistence story - a trained parameter set becomes
          an ordinary checkpoint file that survives restarts, can be copied
          between machines and can be read back by ``Load Parameters``. The node
          also has an output, so it can sit in the middle of a chain: saving does
          not end the graph.
    In:   params (PARAMS) - the parameter set to write; anything a trainer or a
          ``Learnable Parameters`` node publishes.
          filename_prefix (STRING) - output subfolder and file stem, e.g.
          ``"comfydl/parameters"``; ``%date%`` style placeholders are supported,
          and a counter is appended (``parameters_00001_.safetensors``) so an
          earlier save is never overwritten.
    Out:  params (PARAMS) - the same set, passed through unchanged so the graph
          can continue to use it.
          path (STRING) - absolute path of the file that was written; paste it
          into the ``path`` widget of ``Load Parameters`` (or read it from the
          log) to read the set back later.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingSaveParameters",
            display_name="Save Parameters",
            category=CATEGORY,
            description="Saves a parameter set as a .safetensors file in the output folder, and passes it through.",
            is_output_node=True,
            search_aliases=[
                "save", "write", "checkpoint", "safetensors", "export",
                "trainable parameters",
            ],
            inputs=[
                io.Params.Input("params", tooltip="Parameter set to write to disk."),
                io.String.Input(
                    "filename_prefix",
                    default=DEFAULT_FILE_PREFIX,
                    tooltip="Output subfolder and file stem; a counter is appended to keep earlier saves.",
                ),
            ],
            outputs=[
                io.Params.Output(display_name="params"),
                io.String.Output(display_name="path"),
            ],
        )

    @classmethod
    def execute(cls, params, filename_prefix: str = DEFAULT_FILE_PREFIX) -> io.NodeOutput:
        payload = tp.as_parameter_dict(params)
        if not payload:
            raise ValueError(
                "[Network & Layers] Save Parameters received an empty parameter set; "
                "nothing would be written."
            )
        output_path = _save_parameter_file(
            payload, _text(filename_prefix, DEFAULT_FILE_PREFIX)
        )
        print(
            f"[Network & Layers] Save Parameters: wrote {len(payload)} tensor(s) / "
            f"{tp.parameter_count(payload)} value(s) to {output_path}"
        )
        return io.NodeOutput(payload, output_path)


class TrainingLoadParameters(io.ComfyNode):
    """Reads a parameter set back from a ``.safetensors`` file.

    What: the other half of the persistence story, and the way a trained set is
          reused in a later session: the file is read into ordinary parameters
          (float32, ``requires_grad``) so it can warm start a trainer, feed
          ``Parameters to Tensor``, or be re-saved somewhere else.
    In:   path (STRING) - the file to read. A relative path is resolved against
          ComfyUI's *output* folder (which is where ``Save Parameters`` writes),
          an absolute path is used as it is. The default points at the first file
          ``Save Parameters`` writes, so "save, then load" works without typing.
    Out:  params (PARAMS) - every floating point tensor of the file, keyed by its
          key in the file. Integer / non tensor entries are reported and dropped,
          because only floating point tensors can be trained; tensors that are
          not float32 are promoted, also with a report.
    Raises:
        ValueError: when the file does not exist or holds no usable tensor; the
            message carries the resolved absolute path.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingLoadParameters",
            display_name="Load Parameters",
            category=CATEGORY,
            description="Reads a parameter set from a .safetensors file (relative to the output folder, or absolute).",
            search_aliases=[
                "load", "read", "checkpoint", "safetensors", "import",
                "trainable parameters",
            ],
            inputs=[
                io.String.Input(
                    "path",
                    default=DEFAULT_LOAD_PATH,
                    placeholder="e.g. comfydl/parameters_00001_.safetensors",
                    tooltip="File to read; relative to the output folder, or absolute.",
                ),
            ],
            outputs=[io.Params.Output(display_name="params")],
        )

    @classmethod
    def execute(cls, path: str = DEFAULT_LOAD_PATH) -> io.NodeOutput:
        resolved = _resolve_parameter_path(path)
        if not os.path.isfile(resolved):
            raise ValueError(
                f"[Network & Layers] Load Parameters found no file at {resolved}; run "
                "'Save Parameters' first, or fix the 'path' widget (relative paths "
                "start at the output folder)."
            )
        state = comfy.utils.load_torch_file(resolved, safe_load=True)
        payload, dropped, converted = _parameters_from_state(state)
        if dropped:
            _warn(
                f"{len(dropped)} entry / entries in {resolved} are not floating point "
                f"tensors and cannot be trained: "
                f"{tp.parameter_names({name: None for name in dropped}, NAME_PREVIEW)}"
            )
        if converted:
            _warn(
                f"{len(converted)} entry / entries in {resolved} were promoted to "
                f"float32: {tp.parameter_names({name: None for name in converted}, NAME_PREVIEW)}"
            )
        if not payload:
            raise ValueError(
                f"[Network & Layers] Load Parameters found no trainable tensor in "
                f"{resolved}."
            )
        print(
            f"[Network & Layers] Load Parameters: read {len(payload)} tensor(s) / "
            f"{tp.parameter_count(payload)} value(s) from {resolved}"
        )
        return io.NodeOutput(payload)


class TrainingParametersToText(io.ComfyNode):
    """Encodes a parameter set as text, so it can be carried by a widget.

    What: the widget half of the persistence story, and the reason a trained set
          can survive inside a saved ``.json`` workflow without any file: the
          parameters are written into a ``safetensors`` blob and base64 encoded.
          The node is an output node and also pushes the text into its own UI, so
          the value can simply be copied out of the node and pasted into the
          ``text`` widget of a ``Text to Parameters`` node.
    In:   params (PARAMS) - the parameter set to encode.
    Out:  text (STRING) - ``CDLPARAMS1:<base64>``; paste it into ``Text to
          Parameters``. Long sets produce long text - for anything larger than a
          few thousand values prefer ``Save Parameters`` / ``Load Parameters``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingParametersToText",
            display_name="Parameters to Text",
            category=CATEGORY,
            description="Encodes a parameter set into copyable text (CDLPARAMS1:<base64>) that survives inside a workflow file.",
            is_output_node=True,
            search_aliases=["encode", "text", "base64", "copy", "share", "inline"],
            inputs=[io.Params.Input("params", tooltip="Parameter set to encode as text.")],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, params) -> io.NodeOutput:
        payload = tp.as_parameter_dict(params)
        text = tp.encode_parameters(payload)
        if len(text) <= 4000:
            print(
                f"[Network & Layers] Parameters to Text ({len(payload)} tensor(s) / "
                f"{tp.parameter_count(payload)} value(s)):\n{text}"
            )
        else:
            print(
                f"[Network & Layers] Parameters to Text: {len(payload)} tensor(s) / "
                f"{tp.parameter_count(payload)} value(s) encoded into {len(text)} "
                "characters; copy the text from the node's own output box (too long "
                "to print - prefer Save Parameters / Load Parameters for big sets)."
            )
        return io.NodeOutput(text, ui={"text": [text]})


class TrainingTextToParameters(io.ComfyNode):
    """Decodes the text form of a parameter set back into ``PARAMS``.

    What: the reading end of the widget channel. Paste the ``text`` output of a
          ``Parameters to Text`` node into this node's widget - or keep the
          default, which is a valid tiny parameter set - and it comes back as a
          trainable parameter set.
    In:   text (STRING) - ``CDLPARAMS1:<base64>`` as produced by ``Parameters to
          Text``; surrounding whitespace and line breaks are ignored, and the
          header may be omitted when only the base64 body was copied.
    Out:  params (PARAMS) - the decoded parameters, as float32 ``requires_grad``
          tensors keyed by their original names.
    Raises:
        ValueError: when the text is empty, is not valid base64, or does not
            decode into a ``safetensors`` payload; the message says what to paste.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainingTextToParameters",
            display_name="Text to Parameters",
            category=CATEGORY,
            description="Decodes text produced by 'Parameters to Text' back into a trainable parameter set.",
            search_aliases=["decode", "text", "base64", "paste", "inline"],
            inputs=[
                io.String.Input(
                    "text",
                    default=DEFAULT_PARAMS_TEXT,
                    multiline=True,
                    tooltip="Text produced by 'Parameters to Text'; the default is a valid 2x3 'weight'.",
                ),
            ],
            outputs=[io.Params.Output(display_name="params")],
        )

    @classmethod
    def execute(cls, text: str = DEFAULT_PARAMS_TEXT) -> io.NodeOutput:
        try:
            state = tp.decode_parameters(text)
        except ValueError as error:
            raise ValueError(f"[Network & Layers] Text to Parameters: {error}") from error
        payload, dropped, converted = _parameters_from_state(state)
        if dropped:
            _warn(
                f"{len(dropped)} encoded entry / entries are not floating point "
                f"tensors and cannot be trained: "
                f"{tp.parameter_names({name: None for name in dropped}, NAME_PREVIEW)}"
            )
        if not payload:
            raise ValueError(
                "[Network & Layers] Text to Parameters decoded no trainable tensor."
            )
        print(
            f"[Network & Layers] Text to Parameters: decoded {len(payload)} tensor(s) / "
            f"{tp.parameter_count(payload)} value(s)"
            + (f" ({len(converted)} promoted to float32)" if converted else "")
        )
        return io.NodeOutput(payload)


TRAINING_NODES = [
    TrainingParameters,
    TrainingParametersMerge,
    TrainingParametersExtract,
    TrainingOptimizer,
    TrainingLoop,
    TrainingSaveParameters,
    TrainingLoadParameters,
    TrainingParametersToText,
    TrainingTextToParameters,
]


class TrainingExtension(ComfyExtension):
    """Registers the core Training family (parameters, optimizer, trainer, persistence)."""

    @override
    async def get_node_list(cls) -> list[type[io.ComfyNode]]:
        return list(TRAINING_NODES)


async def comfy_entrypoint() -> TrainingExtension:
    return TrainingExtension()
