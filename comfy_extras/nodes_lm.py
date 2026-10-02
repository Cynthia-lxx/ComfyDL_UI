"""Core language-model nodes (reform step 8): spec chain, build, train, generate.

The model half of the language-model pipeline. ComfyUI executes a prompt
inside ``torch.inference_mode()``, so a gradient cannot cross a node boundary
and a language model cannot be assembled as a pure TENSOR dataflow: the
structure is therefore stated as a *spec chain* (``MODELSPEC`` slot, frozen
blueprints, no tensors), materialised by ``Language Model Build`` into a real
``nn.Module`` (``NNMODEL`` slot) and trained inside a single node - the same
technique the step-6 ``Training Loop`` and upstream's ``TrainLoraNode`` use.

* ``Language Model Embedding``      - the first spec link: token embedding
  width, vocabulary size, optional built-in sinusoidal position encoding.
* ``Language Model Transformer Block`` - appends one pre-LN transformer block
  to a chain; stack as many as the widget budget allows.
* ``Language Model Build``          - materialises a chain into a seeded model.
* ``Language Model Train``          - forward + backward + ``optimizer.step()``
  inside one node, on the (context, next-token) samples of ``Sliding Window``;
  returns a *new* trained model, never mutating its input.
* ``Language Model Forward``        - next-token logits of a batch of sequences.
* ``Language Model Generate``       - autoregressive continuation, greedy or
  temperature sampled, seeded locally.
* ``Language Model Save``           - writes the model to ``.safetensors`` with
  the spec chain and the vocabulary in the file's metadata, and passes the
  model through so saving does not end the graph.
* ``Language Model Load``           - reads such a file back: rebuilds the
  blueprint, restores the weights, returns model + vocabulary - a saved model
  talks text again in a fresh session without re-running Vocab Build.

The optimizer settings arrive through the existing ``OPTIMIZER`` slot, so any
``Optimizer`` node drives the trainer; the loss is token-level cross entropy
computed at *every* position (each token predicts its successor, the last
position predicts the wired target), which is what makes a 300-step teaching
run converge.
"""

import copy
import os
import time

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy
import folder_paths
from comfy import lm_protocol as mp
from comfy import loss_preview
from comfy import training_protocol as tp
from comfy_api.latest import ComfyExtension, io

# Absolute import on purpose - the extras loader names modules by file path, so
# a relative import would have no parent package (same convention as the other
# extras files). Reusing the text helpers keeps the token-index contract in one
# place instead of drifting between files. (nodes_text.py is the pre-existing
# Save Text node; the language-model text pipeline lives in nodes_nlp.py.)
from comfy_extras.nodes_nlp import _as_index_tensor, _as_vocab
from comfy_extras.nodes_training import _resolve_parameter_path

CATEGORY = "Network & Layers/Training"

#: Default model width; divisible by the default head count of 4.
DEFAULT_D_MODEL = 32

#: Default FFN width of a block; 4x the model width is the classic ratio.
DEFAULT_D_FFN = 128

#: Above this many iterations the trainer warns: the loop is plain Python.
STEPS_WARN_THRESHOLD = 20000

#: Widget max for a 32-bit seed, matching the training family.
_FF = 0xFFFFFFFF

#: Default output subfolder + file stem of ``Save Language Model``.
DEFAULT_LM_PREFIX = "comfydl/language_models"

#: Default of the ``path`` widget of ``Load Language Model``: the first file a
#: save with :data:`DEFAULT_LM_PREFIX` writes, so a save followed by a load
#: works without anyone typing a file name.
DEFAULT_LM_LOAD_PATH = f"{DEFAULT_LM_PREFIX}_00001_.safetensors"


def _warn(message: str) -> None:
    """Print a short, non-fatal warning (ComfyUI surfaces stdout to the user)."""
    print(f"[Network & Layers] {message}")


def _chain_width(chain) -> int | None:
    """Read the model width of a spec chain, or ``None`` for a non-embedding one."""
    specs = tuple(chain or ())
    return specs[0].d_model if specs and isinstance(specs[0], mp.EmbeddingSpec) else None


def _as_chain(value):
    """Validate a MODELSPEC slot payload."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise ValueError(
        f"this slot needs a spec chain (link it from Language Model Embedding "
        f"/ Transformer Block); got {type(value).__name__}."
    )


def _as_language_model(value) -> mp.LanguageModel:
    """Validate an NNMODEL slot payload with a readable error."""
    if isinstance(value, mp.LanguageModel):
        return value
    raise ValueError(
        f"this slot needs a Language Model (link it from Language Model Build "
        f"or Train); got {type(value).__name__}."
    )


def _as_optimizer(value) -> tp.OptimizerConfig:
    """Coerce the OPTIMIZER slot to a config, warning and defaulting on junk."""
    if isinstance(value, tp.OptimizerConfig):
        return value
    _warn(
        "the 'optimizer' slot did not receive an Optimizer node; using the "
        "default AdamW settings."
    )
    return tp.optimizer_config()


def _targets_for(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Build the aligned ``(N, L)`` target tensor of a next-token batch.

    ``y`` may be the Sliding Window form ``(N,)`` - the token after each
    context - in which case each row's target is its own context shifted left
    with ``y`` appended (every position predicts its successor); or an already
    aligned ``(N, L)`` tensor, used as is.
    """
    if y.dim() == 2 and tuple(y.shape) == tuple(x.shape):
        return y
    if y.dim() == 1 and y.numel() == x.shape[0]:
        # (N, L) targets: x[:, 1:] then the wired next token per row.
        return torch.cat([x[:, 1:], y.unsqueeze(1)], dim=1)
    raise ValueError(
        f"'y' must be the (N,) next-token tensor of the Sliding Window node or "
        f"an (N, L) tensor aligned with 'x' (N={x.shape[0]}, L={x.shape[1]}); "
        f"got shape {tuple(y.shape)}."
    )


class LanguageModelEmbedding(io.ComfyNode):
    """The first link of a language-model spec chain: token embedding.

    What: declares the input half of the model - vocabulary size, model width
          and whether the fixed sinusoidal position encoding is added to the
          embedded tokens (on by default; a stack without recurrence needs
          order information, and the table has no learnable parameters). The
          node outputs the one-link chain that Transformer Block nodes extend.
    In:   spec (MODELSPEC, optional) - an existing chain whose embedding link
          this node replaces (the transformer blocks are kept); leave
          unconnected to start a fresh chain.
          vocab (VOCAB, optional) - link from Vocab Build to read
          ``vocab_size`` automatically; overrides the widget when connected.
          vocab_size (INT) - vocabulary size when ``vocab`` is unconnected
          (default 16). Must match the vocabulary the model will be trained
          on, or the trained head addresses the wrong token set.
          d_model (INT) - model width (default 32); every block of the chain
          uses this width and the head projects back to ``vocab_size``.
          include_position (BOOLEAN) - add the sinusoidal position encoding
          (default on).
    Out:  spec (MODELSPEC) - the chain starting with this embedding link.
          d_model (INT) - the declared width, for wiring into other nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelEmbedding",
            display_name="Language Model Embedding",
            category=CATEGORY,
            description="First spec link of a language model: token embedding width, vocabulary size and the optional sinusoidal position encoding.",
            search_aliases=["language model", "embedding", "spec", "transformer", "vocab", "position", "lm"],
            inputs=[
                io.ModelSpec.Input(
                    "spec",
                    optional=True,
                    tooltip="Optional existing chain whose embedding link is replaced (blocks are kept); leave unconnected to start a fresh chain.",
                ),
                io.Vocab.Input(
                    "vocab",
                    optional=True,
                    tooltip="Link from Vocab Build to read vocab_size automatically; overrides the widget.",
                ),
                io.Int.Input(
                    "vocab_size",
                    default=16,
                    min=2,
                    max=1000000,
                    step=1,
                    tooltip="Vocabulary size (the VOCAB link overrides this); the output head has this many logits.",
                ),
                io.Int.Input(
                    "d_model",
                    default=DEFAULT_D_MODEL,
                    min=2,
                    max=4096,
                    step=1,
                    tooltip="Model width; every transformer block of the chain shares it.",
                ),
                io.Boolean.Input(
                    "include_position",
                    default=True,
                    tooltip="Add the fixed sinusoidal position encoding after the embedding (no learnable parameters).",
                ),
            ],
            outputs=[
                io.ModelSpec.Output(display_name="spec"),
                io.Int.Output(display_name="d_model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        spec=None,
        vocab=None,
        vocab_size: int = 16,
        d_model: int = DEFAULT_D_MODEL,
        include_position: bool = True,
    ) -> io.NodeOutput:
        if vocab is not None:
            size = _as_vocab(vocab).size
        else:
            size = int(vocab_size)
        if size < 2:
            raise ValueError(
                f"vocab_size must be >= 2 (a one-token vocabulary has nothing "
                f"to predict); got {size}."
            )
        width = int(d_model)
        if width < 2:
            raise ValueError(f"d_model must be >= 2; got {width}.")
        link = mp.EmbeddingSpec(size, width, include_position=bool(include_position))
        incoming = tuple(spec) if isinstance(spec, (tuple, list)) and spec else ()
        if incoming:
            if not isinstance(incoming[0], mp.EmbeddingSpec):
                raise ValueError(
                    "the wired chain does not start with an embedding link; a "
                    "transformer block cannot come first. Feed the chain "
                    "through the Embedding node, not around it."
                )
            return io.NodeOutput((link, *incoming[1:]), width)
        return io.NodeOutput((link,), width)


class LanguageModelTransformerBlock(io.ComfyNode):
    """Append one pre-LN transformer block to a language-model spec chain.

    What: one ``x + attn(LN(x))`` then ``x + ffn(LN(x))`` block - causal
          self-attention over ``num_heads`` heads plus a two-layer FFN with
          ``d_ffn`` hidden units. The width is *read from the incoming chain*
          (the embedding link declares it), so a mismatched block is
          impossible to wire. Chain as many blocks as wanted by feeding each
          node's ``spec`` output into the next one's ``spec`` input; the model
          is as deep as the chain.
    In:   spec (MODELSPEC) - the chain so far; must start with a Language
          Model Embedding link.
          num_heads (INT) - attention heads (default 4); must divide the chain
          width.
          d_ffn (INT) - FFN hidden width (default 128, i.e. 4x the default
          model width).
          activation (COMBO) - between the FFN linear layers, ``relu`` or
          ``gelu``.
          dropout (FLOAT) - dropout probability inside the block; active only
          while training (the Train node's job), 0 disables.
    Out:  spec (MODELSPEC) - the chain with this block appended.
          d_model (INT) - the chain width, unchanged (pass-through).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelTransformerBlock",
            display_name="Language Model Transformer Block",
            category=CATEGORY,
            description="Appends one pre-LN transformer block (causal self-attention + FFN) to a language-model spec chain; the width follows the embedding link.",
            search_aliases=["transformer", "block", "spec", "language model", "attention", "ffn", "residual", "lm"],
            inputs=[
                io.ModelSpec.Input(
                    "spec",
                    tooltip="The chain so far; link from Language Model Embedding (or a previous Transformer Block).",
                ),
                io.Int.Input(
                    "num_heads",
                    default=4,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Attention heads; must divide the model width read from the chain.",
                ),
                io.Int.Input(
                    "d_ffn",
                    default=DEFAULT_D_FFN,
                    min=2,
                    max=16384,
                    step=1,
                    tooltip="Hidden width of the feed-forward half; 4x the model width is the classic ratio.",
                ),
                io.Combo.Input(
                    "activation",
                    options=list(mp.ACTIVATION_OPTIONS),
                    default="relu",
                    tooltip="Activation between the two FFN linear layers.",
                ),
                io.Float.Input(
                    "dropout",
                    default=0.0,
                    min=0.0,
                    max=0.9,
                    step=0.05,
                    tooltip="Dropout inside the block (attention weights and FFN hidden); active only while training.",
                ),
            ],
            outputs=[
                io.ModelSpec.Output(display_name="spec"),
                io.Int.Output(display_name="d_model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        spec,
        num_heads: int = 4,
        d_ffn: int = DEFAULT_D_FFN,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> io.NodeOutput:
        chain = _as_chain(spec)
        width = _chain_width(chain)
        if width is None:
            raise ValueError(
                "the incoming chain does not start with a Language Model "
                "Embedding link; a transformer block must follow one."
            )
        block = mp.TransformerBlockSpec(
            d_model=width,
            num_heads=max(1, int(num_heads)),
            d_ffn=max(2, int(d_ffn)),
            activation=activation,
            dropout=min(max(float(dropout), 0.0), 0.9),
        )
        return io.NodeOutput((*chain, block), width)


class LanguageModelBuild(io.ComfyNode):
    """Materialise a spec chain into a real, seeded language model.

    What: the bridge from the blueprint to the ``NNMODEL`` slot. Validates the
          chain (embedding first, consistent width, heads divide the width),
          builds the module and initialises it deterministically from ``seed``
          - Xavier-uniform linear weights, zero biases, N(0, 0.01) embeddings -
          inside a save-and-restore RNG block, so the same chain and seed
          always produce the same weights and the process RNG stays untouched.
    In:   spec (MODELSPEC) - the finished chain (embedding link + any number
          of transformer blocks; zero blocks make a bag-of-contexts linear
          model, which is itself an instructive baseline).
          seed (INT) - the initialisation seed.
    Out:  model (NNMODEL) - the untrained model, ready for Language Model
          Train / Forward / Generate.
          params (INT) - the total trainable parameter count.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelBuild",
            display_name="Language Model Build",
            category=CATEGORY,
            description="Materialises a spec chain into a seeded nn.Module language model and reports its parameter count.",
            search_aliases=["build", "materialise", "language model", "spec", "xavier", "init", "lm"],
            inputs=[
                io.ModelSpec.Input(
                    "spec",
                    tooltip="The spec chain to materialise; link from Language Model Embedding / Transformer Block.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=_FF,
                    step=1,
                    control_after_generate=True,
                    tooltip="Initialisation seed; the same chain and seed always build the same weights.",
                ),
            ],
            outputs=[
                io.NNModel.Output(display_name="model"),
                io.Int.Output(display_name="params"),
            ],
        )

    @classmethod
    def execute(cls, spec, seed: int = 0) -> io.NodeOutput:
        model = mp.build_model(_as_chain(spec), seed=int(seed))
        return io.NodeOutput(model, mp.parameter_count(model))


class LanguageModelTrain(io.ComfyNode):
    """Train a language model on next-token samples, inside the node.

    What: the training closure. ComfyUI executes every node under
          ``torch.inference_mode()`` and caches the result, so a gradient
          cannot survive a node boundary: this node therefore runs forward,
          backward and ``optimizer.step()`` itself, for ``steps`` iterations,
          on a *deep copy* of the input model - the cached input stays
          untouched and the node returns a new trained model.           The loss is
          token-level cross entropy computed at every position: each context
          token predicts its successor and the last position predicts the
          wired ``y``, so one sample trains ``window`` predictions at once.
          Three optional per-iteration pieces: a wired ``SCHEDULER`` is stepped
          after every ``optimizer.step()`` (``ReduceLROnPlateau`` with the
          smoothed step loss, the others without arguments), gradient clipping
          travels in the ``OPTIMIZER`` config (``grad_clip_norm`` /
          ``grad_clip_value``, 0 = off) and runs right before the step, and
          early stopping watches the moving average of the step loss (window 8):
          once ``early_stop_patience`` steps pass without an improvement of
          ``early_stop_min_delta``, the copy's parameters are rolled back to the
          best point and the run ends there.
          While training, the node also drives ComfyUI's live preview channel:
          the cross-entropy curve recorded so far is rendered (rate-limited to
          one frame per ~0.5 s, always a final frame at the end) and shown
          under the node - the same side channel the KSampler latent preview
          uses.
    In:   model (NNMODEL) - link from Language Model Build (or another Train
          node, to continue training).
          x (TENSOR) - ``(samples, window)`` long contexts; link from Sliding
          Window's ``x``.
          y (TENSOR) - ``(samples,)`` long next tokens; link from Sliding
          Window's ``y``. An ``(N, L)`` tensor aligned with ``x`` is also
          accepted.
          optimizer (OPTIMIZER) - settings published by an Optimizer node,
          including the gradient clipping widgets.
          scheduler (SCHEDULER, optional) - settings published by an
          ``LR Scheduler`` node. Unconnected means a constant learning rate,
          exactly like before this slot existed.
          steps (INT) - optimizer steps (default 300; each step is one
          forward + backward + update on the chosen batch).
          batch_size (INT) - samples per step; ``0`` (default) uses the whole
          dataset every step, the most stable choice for small teaching data.
          seed (INT) - seeds the dropout draws and the batch shuffling, so the
          same inputs always produce the same run.
          early_stop_patience (INT) - stop after this many steps without an
          improvement of the smoothed loss; 0 (the default) disables early
          stopping.
          early_stop_min_delta (FLOAT) - the smallest improvement of the
          smoothed loss that resets the patience counter.
    Out:  model (NNMODEL) - the trained copy, in eval mode (dropout off); with
          early stopping it holds the parameters of the *best* step, not the
          last one.
          loss (FLOAT) - the last entry of the (possibly truncated) loss
          history.
          loss_history (TENSOR) - 1-D, one entry per executed step; with early
          stopping it is truncated to the best step. The convergence curve,
          ready for any of the visualisation nodes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelTrain",
            display_name="Language Model Train",
            category=CATEGORY,
            description="Trains a language model on (context, next-token) samples inside the node: forward + backward + optimizer.step per step, returning a new trained model and the loss history.",
            search_aliases=["train", "language model", "fit", "gradient descent", "backward", "cross entropy", "lm", "loss"],
            inputs=[
                io.NNModel.Input(
                    "model",
                    tooltip="The model to train; link from Language Model Build (or a previous Train).",
                ),
                io.Tensor.Input(
                    "x",
                    tooltip="(samples, window) long contexts; link from Sliding Window's x.",
                ),
                io.Tensor.Input(
                    "y",
                    tooltip="(samples,) long next tokens; link from Sliding Window's y.",
                ),
                io.Optimizer.Input(
                    "optimizer",
                    tooltip="Optimizer settings, linked from an Optimizer node.",
                ),
                io.Scheduler.Input(
                    "scheduler",
                    optional=True,
                    tooltip="Optional LR scheduler settings from an LR Scheduler node; unconnected = constant learning rate.",
                ),
                io.Int.Input(
                    "steps",
                    default=300,
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
                    max=_FF,
                    step=1,
                    control_after_generate=True,
                    tooltip="Seeds the dropout draws and the batch shuffling; the same seed reproduces the same run.",
                ),
                io.Int.Input(
                    "early_stop_patience",
                    default=0,
                    min=0,
                    max=100000,
                    step=1,
                    tooltip="Stop after this many steps without an improvement of the smoothed loss; 0 disables early stopping.",
                ),
                io.Float.Input(
                    "early_stop_min_delta",
                    default=1e-4,
                    min=0.0,
                    max=1.0,
                    step=1e-4,
                    tooltip="Smallest improvement of the smoothed loss that resets the patience counter.",
                ),
            ],
            outputs=[
                io.NNModel.Output(display_name="model"),
                io.Float.Output(display_name="loss"),
                io.Tensor.Output(display_name="loss_history"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer,
        scheduler=None,
        steps: int = 300,
        batch_size: int = 0,
        seed: int = 0,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 1e-4,
    ) -> io.NodeOutput:
        source = _as_language_model(model)
        config = _as_optimizer(optimizer)
        if scheduler is not None and not isinstance(scheduler, tp.SchedulerConfig):
            _warn(
                "the 'scheduler' slot did not receive an LR Scheduler node; "
                "keeping the learning rate constant."
            )
            scheduler = None

        if x.dim() != 2:
            raise ValueError(
                f"'x' must be a 2-D (samples, window) tensor; got shape {tuple(x.shape)}."
            )
        contexts = _as_index_tensor(x, "x").reshape(x.shape)
        samples = contexts.shape[0]
        targets_flat = _as_index_tensor(y, "y")

        # Vocab guard: indices outside the model's head are a wiring error.
        head = source.vocab_size
        if int(contexts.max().item()) >= head:
            raise ValueError(
                f"'x' holds a token index >= the model's vocabulary size "
                f"({head}); the vocab and the model's embedding link do not "
                "match."
            )
        targets = _targets_for(contexts, targets_flat)
        if int(targets.max().item()) >= head:
            raise ValueError(
                f"'y' holds a token index >= the model's vocabulary size "
                f"({head}); the vocab and the model's embedding link do not "
                "match."
            )

        iterations = max(1, int(steps))
        if iterations > STEPS_WARN_THRESHOLD:
            _warn(
                f"{iterations} steps were requested; the loop is plain Python and "
                "shares the machine with the rest of the graph."
            )
        per_step = int(batch_size)
        if per_step <= 0 or per_step >= samples:
            if per_step > samples:
                _warn(
                    f"batch_size {per_step} is larger than the {samples} "
                    "sample(s); using the whole dataset every step."
                )
            per_step = 0

        history: list[torch.Tensor] = []
        # Everything below runs outside ComfyUI's inference_mode: an inference
        # tensor cannot take part in a backward pass, and the copy's parameters
        # have to stay ordinary (grad carrying) tensors.
        with torch.inference_mode(False):
            # clone() of an inference tensor with inference mode disabled is a
            # normal tensor again - the same escape hatch the Training Loop uses.
            trainee = copy.deepcopy(source)
            inputs = contexts.clone()
            labels = targets.clone()
            shuffler = torch.Generator()
            shuffler.manual_seed(int(seed))
            with tp.seeded_rng(int(seed)):
                trainer = tp.build_optimizer(config, trainee.parameters())
                trainee.train()
                # Optional LR schedule, built against the real step count so a
                # follow-the-trainer horizon (t_max 0 / OneCycleLR) is exact.
                lr_schedule = None
                if scheduler is not None:
                    lr_schedule, schedule_needs_metric = tp.build_scheduler(
                        scheduler, trainer, iterations
                    )
                # Optional early stopping on the smoothed loss; the tracker owns
                # no tensors, the snapshot / rollback below do.
                stopper = tp.EarlyStopTracker(early_stop_patience, early_stop_min_delta)
                best_state: dict[str, torch.Tensor] | None = None
                best_length = 0
                stopped_at: int | None = None
                # Live progress: one update per optimizer step.  The ProgressBar
                # hook binds itself to the executing node and doubles as the
                # interrupt check, so a long run stays cancellable from the UI.
                # The third update_absolute argument attaches the live
                # loss-curve preview (rate-limited inside LossCurvePreviewer).
                pbar = comfy.utils.ProgressBar(iterations)
                curve_previewer = loss_preview.LossCurvePreviewer(title="cross entropy")
                for step in range(iterations):
                    if per_step:
                        order = torch.randperm(samples, generator=shuffler)[:per_step]
                        batch_x = inputs.index_select(0, order)
                        batch_y = labels.index_select(0, order)
                    else:
                        batch_x, batch_y = inputs, labels
                    logits = trainee(batch_x)
                    step_loss = F.cross_entropy(
                        logits.reshape(-1, trainee.vocab_size), batch_y.reshape(-1)
                    )
                    trainer.zero_grad(set_to_none=True)
                    step_loss.backward()
                    if config.grad_clip_norm > 0.0 or config.grad_clip_value > 0.0:
                        tp.clip_gradients(config, trainee.parameters())
                    trainer.step()
                    if lr_schedule is not None:
                        if schedule_needs_metric:
                            lr_schedule.step(float(step_loss.detach()))
                        else:
                            lr_schedule.step()
                    history.append(step_loss.detach().reshape(()))
                    if stopper.update(step, float(step_loss.detach())):
                        stopped_at = step
                        break
                    if stopper.improved:
                        best_state = tp.snapshot_state(trainee)
                        best_length = len(history)
                    frame = curve_previewer.record(
                        float(step_loss.detach()), force=(step + 1 == iterations)
                    )
                    if frame is not None:
                        pbar.update_absolute(step + 1, iterations, frame)
                    else:
                        pbar.update(1)
                if stopped_at is not None and best_state is not None:
                    # Roll back to the best point and cut the history there, so
                    # the outputs are what "the best model of this run" means.
                    tp.restore_state(trainee, best_state)
                    del history[best_length:]
                    _warn(
                        f"Language Model Train: early stop at step {stopped_at + 1} "
                        f"- {stopper.describe_stop()}"
                    )
                trainer.zero_grad(set_to_none=True)
            trainee.eval()
            for parameter in trainee.parameters():
                parameter.grad = None

        return io.NodeOutput(
            trainee,
            float(history[-1].item()) if history else float("nan"),
            torch.stack(history) if history else torch.zeros(0),
        )


class LanguageModelForward(io.ComfyNode):
    """Next-token logits of a language model on a batch of sequences.

    What: a pure inference pass - the model runs in eval mode (dropout off)
          under ``no_grad`` and maps every sequence to its per-position logits.
          Useful for inspecting what the model believes (wire the logits into
          the argmax / softmax tensor nodes), or for a hand-rolled decoding
          loop of your own design.
    In:   model (NNMODEL) - link from Language Model Build / Train.
          ids (TENSOR) - token indices: a 1-D stream (treated as one
          sequence) or a 2-D ``(batch, seq_len)`` batch.
    Out:  logits (TENSOR) - ``(batch, seq_len, vocab_size)`` float32; entry
          ``[..., t, :]`` is the distribution of the token *after* position
          ``t``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelForward",
            display_name="Language Model Forward",
            category=CATEGORY,
            description="Runs a language model in eval mode and returns the per-position next-token logits of a batch of token-index sequences.",
            search_aliases=["forward", "inference", "logits", "language model", "predict", "lm"],
            inputs=[
                io.NNModel.Input(
                    "model",
                    tooltip="The model to run; link from Language Model Build / Train.",
                ),
                io.Tensor.Input(
                    "ids",
                    tooltip="Token indices: a 1-D stream or a 2-D (batch, seq_len) batch.",
                ),
            ],
            outputs=[io.Tensor.Output(display_name="logits")],
        )

    @classmethod
    def execute(cls, model, ids: torch.Tensor) -> io.NodeOutput:
        trainee = _as_language_model(model)
        if ids.dim() <= 1:
            batch = _as_index_tensor(ids, "ids").unsqueeze(0)
        else:
            batch = _as_index_tensor(ids, "ids").reshape(-1, ids.shape[-1])
        was_training = trainee.training
        trainee.eval()
        try:
            with torch.no_grad():
                logits = trainee(batch)
        finally:
            trainee.train(was_training)
        return io.NodeOutput(logits)


class LanguageModelGenerate(io.ComfyNode):
    """Autoregressive continuation: the "talk to your model" node.

    What: repeatedly takes the model's next-token distribution and appends the
          chosen token - ``num_tokens`` times - starting from a prefix. The
          prefix comes from ``prefix_ids`` when wired, otherwise from the
          ``prefix`` text through ``vocab`` (encode). ``temperature`` controls
          the sampler: 0 (or any value <= 0) is deterministic greedy argmax,
          1.0 samples the plain softmax, below 1 sharpens toward the argmax,
          above 1 flattens toward uniform. The sampling runs on a local
          ``torch.Generator`` seeded by ``seed`` - the same inputs always
          reproduce the same continuation.
          While generating, the node also drives ComfyUI's live preview
          channel: the text generated so far is rendered onto a small card
          (rate-limited to one frame per ~0.5 s, always a final frame at the
          end) and shown under the node - the same side channel the KSampler
          latent preview uses. Needs a wired ``vocab`` to decode; without one
          the preview shows the raw token indices instead.
    In:   model (NNMODEL) - link from Language Model Train (a trained model
          generates text; an untrained one generates noise).
          vocab (VOCAB, optional) - link from Vocab Build to read the ``prefix``
          text and decode the ``text`` output. Without it the node still
          generates - indices in, indices out.
          prefix (STRING) - the prompt text, encoded through ``vocab``; used
          when ``prefix_ids`` is unconnected.
          prefix_ids (TENSOR, optional) - token indices to start from;
          overrides ``prefix`` when wired.
          num_tokens (INT) - how many tokens to generate (default 16).
          temperature (FLOAT) - sampling temperature; 0 = greedy (default
          1.0).
          seed (INT) - seed of the sampling generator.
    Out:  ids (TENSOR) - 1-D long: the prefix followed by the generated
          tokens.
          text (STRING) - the decoded continuation (prefix included); empty
          when ``vocab`` is unconnected.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelGenerate",
            display_name="Language Model Generate",
            category=CATEGORY,
            description="Autoregressive continuation of a prefix: greedy or temperature-sampled next tokens, seeded locally, with optional text decode through a vocabulary.",
            search_aliases=["generate", "autoregressive", "sample", "language model", "text", "chat", "continue", "lm"],
            inputs=[
                io.NNModel.Input(
                    "model",
                    tooltip="The model to sample from; link from Language Model Train / Build.",
                ),
                io.Vocab.Input(
                    "vocab",
                    optional=True,
                    tooltip="Optional vocabulary: encodes the prefix text and decodes the generated text.",
                ),
                io.String.Input(
                    "prefix",
                    default="the ",
                    placeholder="the prompt text (encoded through vocab)",
                    tooltip="The prompt text; used when prefix_ids is unconnected.",
                ),
                io.Tensor.Input(
                    "prefix_ids",
                    optional=True,
                    tooltip="Optional token indices to start from; overrides the prefix text when wired.",
                ),
                io.Int.Input(
                    "num_tokens",
                    default=16,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="How many tokens to generate.",
                ),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip="Sampling temperature: 0 = greedy argmax, 1.0 = plain softmax, <1 sharper, >1 flatter.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=_FF,
                    step=1,
                    control_after_generate=True,
                    tooltip="Seed of the sampling generator; the same seed and prefix reproduce the same continuation.",
                ),
            ],
            outputs=[
                io.Tensor.Output(display_name="ids"),
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        vocab=None,
        prefix: str = "the ",
        prefix_ids: torch.Tensor | None = None,
        num_tokens: int = 16,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> io.NodeOutput:
        trainee = _as_language_model(model)

        if prefix_ids is not None:
            table = _as_vocab(vocab) if vocab is not None else None
            start = _as_index_tensor(prefix_ids, "prefix_ids").tolist()
        elif vocab is not None:
            table = _as_vocab(vocab)
            start = table.encode("" if prefix is None else str(prefix))
        else:
            raise ValueError(
                "the node needs either a wired 'prefix_ids' tensor or a linked "
                "'vocab' to encode the 'prefix' text; neither was found."
            )

        head = trainee.vocab_size
        if start and max(start) >= head:
            raise ValueError(
                f"the prefix holds a token index >= the model's vocabulary size "
                f"({head}); the vocab and the model's embedding link do not match."
            )

        count = max(1, int(num_tokens))
        # Live progress: one update per generated token.  The ProgressBar hook
        # binds itself to the executing node and doubles as the interrupt check
        # (model_management.throw_exception_if_processing_interrupted), so a
        # long generation stays cancellable from the UI.  The preview triple
        # attaches the live text card: the tokens generated so far, decoded
        # and rendered (rate-limited; the last token always renders, so the
        # final frame is the complete text).
        pbar = comfy.utils.ProgressBar(count)
        prefix_text = table.decode(start) if table is not None else ""
        generated: list[int] = []
        last_render = 0.0

        def _snapshot(done: int, force: bool = False) -> tuple | None:
            nonlocal last_render
            now = time.perf_counter()
            if not force and now - last_render < loss_preview.MIN_RENDER_INTERVAL:
                return None
            last_render = now
            if table is not None:
                body = prefix_text + table.decode(generated)
            else:
                body = " ".join(str(index) for index in generated)
            return loss_preview.render_text_snapshot(
                body, title=f"generated {done}/{count}"
            )

        def _on_token(token_id: int) -> None:
            generated.append(int(token_id))
            done = len(generated)
            frame = _snapshot(done, force=(done == count))
            if frame is not None:
                pbar.update_absolute(done, count, frame)
            else:
                pbar.update_absolute(done, count)

        ids = mp.generate_tokens(
            trainee,
            start,
            num_tokens=count,
            temperature=float(temperature),
            seed=int(seed),
            on_token=_on_token,
        )
        text = table.decode(ids.tolist()) if table is not None else ""
        return io.NodeOutput(ids, text)


def _save_lm_file(model: mp.LanguageModel, vocab: mp.Vocab, filename_prefix: str) -> str:
    """Write a language model to a uniquely named ``.safetensors`` in ``output/``.

    The weights go in as an ordinary ``state_dict``; the spec chain and the
    vocabulary travel in the file's metadata (JSON, version-tagged), so a
    later ``Load Language Model`` rebuilds the exact model without the graph
    re-stating anything.

    Returns:
        The absolute path that was written.
    """
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, _subfolder, _prefix = (
        folder_paths.get_save_image_path(filename_prefix, output_dir)
    )
    os.makedirs(full_output_folder, exist_ok=True)
    output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.safetensors")
    comfy.utils.save_torch_file(
        model.state_dict(),
        output_path,
        metadata={
            mp.LM_META_FORMAT: mp.LM_FILE_FORMAT,
            mp.LM_META_SPEC: mp.spec_chain_to_json(mp.spec_chain_from_model(model)),
            mp.LM_META_VOCAB: mp.vocab_to_json(vocab),
        },
    )
    return output_path


class LanguageModelSave(io.ComfyNode):
    """Persist a language model to a ``.safetensors`` file, vocabulary and all.

    What: the file half of the language-model persistence story - the trained
          weights are written as an ordinary checkpoint that survives
          restarts and can be copied between machines, and the model's
          blueprint (spec chain) plus its vocabulary travel in the same
          file's metadata. ``Load Language Model`` can therefore rebuild the
          exact model and talk text again without re-running Vocab Build or
          re-wiring the spec chain. The node passes the model through, so
          saving does not end the graph.
    In:   model (NNMODEL) - the model to write; link from Language Model Train
          (an untrained Build output saves too - it is just a checkpoint of
          the initial weights).
          vocab (VOCAB) - the vocabulary the model was trained with; it is
          stored in the file and read back by Load, so generation in a later
          session works from text prompt to text reply.
          filename_prefix (STRING) - output subfolder and file stem, e.g.
          ``"comfydl/language_models"``; a counter is appended
          (``language_models_00001_.safetensors``) so an earlier save is
          never overwritten.
    Out:  model (NNMODEL) - the same model, passed through unchanged so the
          graph can continue (e.g. straight into Language Model Generate).
          path (STRING) - absolute path of the file that was written; paste it
          into the ``path`` widget of ``Load Language Model`` to read the
          model back later.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelSave",
            display_name="Save Language Model",
            category=CATEGORY,
            description="Saves a language model as a .safetensors file in the output folder (spec chain and vocabulary in the metadata), and passes the model through.",
            is_output_node=True,
            search_aliases=[
                "save", "write", "checkpoint", "safetensors", "export",
                "language model", "lm",
            ],
            inputs=[
                io.NNModel.Input(
                    "model",
                    tooltip="The model to write; link from Language Model Train / Build.",
                ),
                io.Vocab.Input(
                    "vocab",
                    tooltip="The vocabulary to store alongside the weights; Load reads it back so a saved model talks text again.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default=DEFAULT_LM_PREFIX,
                    tooltip="Output subfolder and file stem; a counter is appended to keep earlier saves.",
                ),
            ],
            outputs=[
                io.NNModel.Output(display_name="model"),
                io.String.Output(display_name="path"),
            ],
        )

    @classmethod
    def execute(
        cls, model, vocab, filename_prefix: str = DEFAULT_LM_PREFIX
    ) -> io.NodeOutput:
        trainee = _as_language_model(model)
        table = _as_vocab(vocab)
        prefix = "" if filename_prefix is None else str(filename_prefix).strip()
        output_path = _save_lm_file(trainee, table, prefix or DEFAULT_LM_PREFIX)
        print(
            f"[Network & Layers] Save Language Model: wrote "
            f"{mp.parameter_count(trainee)} parameter value(s) / "
            f"{table.size} vocab token(s) to {output_path}"
        )
        return io.NodeOutput(trainee, output_path)


class LanguageModelLoad(io.ComfyNode):
    """Read a language model back from a ``.safetensors`` file.

    What: the other half of the persistence story, and the way a trained model
          is reused in a later session: the file's metadata states the
          blueprint (spec chain) and the vocabulary, the node rebuilds the
          model from them and restores the weights with a strict
          ``load_state_dict`` - any mismatch between the file and its own
          blueprint is a hard error, so what comes out is bit-for-bit the
          model that was saved. The returned model is in eval mode (dropout
          off), ready for Language Model Forward / Generate or further
          Language Model Train (warm start).
    In:   path (STRING) - the file to read. A relative path is resolved
          against ComfyUI's *output* folder (which is where Save Language
          Model writes), an absolute path is used as it is. The default
          points at the first file Save writes, so "save, then load" works
          without typing.
    Out:  model (NNMODEL) - the restored model, in eval mode.
          vocab (VOCAB) - the vocabulary stored with the model; wire it into
          Language Model Generate to prompt and decode with text.
          params (INT) - the total trainable parameter count.
    Raises:
        ValueError: when the file does not exist, was not written by Save
            Language Model (missing or unknown format tag), or its metadata
            does not describe a valid spec chain / vocabulary.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LanguageModelLoad",
            display_name="Load Language Model",
            category=CATEGORY,
            description="Reads a language model from a .safetensors file written by Save Language Model (relative to the output folder, or absolute), returning model + vocabulary.",
            search_aliases=[
                "load", "read", "checkpoint", "safetensors", "import",
                "language model", "lm",
            ],
            inputs=[
                io.String.Input(
                    "path",
                    default=DEFAULT_LM_LOAD_PATH,
                    placeholder=f"e.g. {DEFAULT_LM_LOAD_PATH}",
                    tooltip="File to read; relative to the output folder, or absolute.",
                ),
            ],
            outputs=[
                io.NNModel.Output(display_name="model"),
                io.Vocab.Output(display_name="vocab"),
                io.Int.Output(display_name="params"),
            ],
        )

    @classmethod
    def execute(cls, path: str = DEFAULT_LM_LOAD_PATH) -> io.NodeOutput:
        resolved = _resolve_parameter_path(path)
        if not os.path.isfile(resolved):
            raise ValueError(
                f"[Network & Layers] Load Language Model found no file at "
                f"{resolved}; run 'Save Language Model' first, or fix the "
                "'path' widget (relative paths start at the output folder)."
            )
        state, metadata = comfy.utils.load_torch_file(
            resolved, safe_load=True, return_metadata=True
        )
        metadata = metadata or {}
        if metadata.get(mp.LM_META_FORMAT) != mp.LM_FILE_FORMAT:
            raise ValueError(
                f"[Network & Layers] {resolved} was not written by 'Save "
                "Language Model' (missing or unknown format tag "
                f"{mp.LM_FILE_FORMAT!r}); the node cannot rebuild the model."
            )
        try:
            chain = mp.spec_chain_from_json(metadata[mp.LM_META_SPEC])
            table = mp.vocab_from_json(metadata[mp.LM_META_VOCAB])
        except KeyError as missing:
            raise ValueError(
                f"[Network & Layers] {resolved} is missing the {missing} "
                "metadata entry; it was not written by 'Save Language Model'."
            ) from missing
        if chain[0].vocab_size != table.size:
            raise ValueError(
                f"[Network & Layers] {resolved} is inconsistent: its blueprint "
                f"declares {chain[0].vocab_size} vocabulary entries but its "
                f"vocabulary holds {table.size}; the file is corrupt."
            )
        with torch.inference_mode(False):
            model = mp.build_model(chain)
            model.load_state_dict(state, strict=True)
            model.eval()
        print(
            f"[Network & Layers] Load Language Model: restored "
            f"{mp.parameter_count(model)} parameter value(s) / {table.size} "
            f"vocab token(s) from {resolved}"
        )
        return io.NodeOutput(model, table, mp.parameter_count(model))


#: Every node this module registers, in node-library order.
LM_NODES: list[type[io.ComfyNode]] = [
    LanguageModelEmbedding,
    LanguageModelTransformerBlock,
    LanguageModelBuild,
    LanguageModelTrain,
    LanguageModelForward,
    LanguageModelGenerate,
    LanguageModelSave,
    LanguageModelLoad,
]


class LanguageModelExtension(ComfyExtension):
    """Registers the core language-model node family."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(LM_NODES)


async def comfy_entrypoint() -> LanguageModelExtension:
    return LanguageModelExtension()
