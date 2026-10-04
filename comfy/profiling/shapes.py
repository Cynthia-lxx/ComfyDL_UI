"""Value objects for the profiling graph (``comfy.profiling``).

The estimators propagate light-weight *values* (not tensors) across the
workflow graph: shapes, vocab sizes, spec chains, plain numbers. Everything
here is plain data - no torch, no ComfyUI machinery - so the whole estimation
engine stays importable and testable on its own, exactly like the other
protocol modules under ``comfy/``.

``Unknown`` is a first-class value: an output that cannot be statically
estimated propagates as Unknown with a human-readable reason, and the report
says "unknown" instead of guessing.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

#: Bytes per element of the dtypes the estimated nodes move around.
FLOAT32_BYTES = 4
INT64_BYTES = 8


class EstValue:
    """Base class of every value that travels between estimators."""

    def describe(self) -> str:
        return type(self).__name__


@dataclasses.dataclass(frozen=True)
class Unknown(EstValue):
    """An output that cannot be statically estimated, with the reason why."""

    reason: str = "not statically known"

    def describe(self) -> str:
        return f"unknown ({self.reason})"


@dataclasses.dataclass(frozen=True)
class IntVal(EstValue):
    """A plain integer output (vocab size, d_model, parameter count, ...)."""

    value: Optional[int] = None

    def describe(self) -> str:
        return "unknown int" if self.value is None else str(self.value)


@dataclasses.dataclass(frozen=True)
class TextVal(EstValue):
    """A text output; only its length matters for estimation."""

    length: Optional[int] = None

    def describe(self) -> str:
        return "unknown text" if self.length is None else f"{self.length} char(s)"


@dataclasses.dataclass(frozen=True)
class TensorVal(EstValue):
    """A tensor output: static shape plus element size.

    A dimension of ``None`` means "not statically known" (``nbytes`` then
    returns ``None``; the consumer decides whether an assumption applies).
    """

    shape: tuple = ()
    itemsize: int = FLOAT32_BYTES
    dtype: str = "float32"

    def numel(self) -> Optional[int]:
        count = 1
        for dim in self.shape:
            if dim is None:
                return None
            count *= int(dim)
        return count

    def nbytes(self) -> Optional[int]:
        count = self.numel()
        return None if count is None else count * self.itemsize

    def dim(self, index: int) -> Optional[int]:
        if -len(self.shape) <= index < len(self.shape):
            value = self.shape[index]
            return None if value is None else int(value)
        return None

    def describe(self) -> str:
        dims = ", ".join("?" if d is None else str(d) for d in self.shape)
        return f"({dims}) {self.dtype}"


@dataclasses.dataclass(frozen=True)
class VocabVal(EstValue):
    """A ``VOCAB`` payload: its size and tokenisation level."""

    size: Optional[int] = None
    level: str = "char"

    def describe(self) -> str:
        size = "?" if self.size is None else str(self.size)
        return f"vocab[{size}] {self.level}"


@dataclasses.dataclass(frozen=True)
class OptimizerVal(EstValue):
    """An ``OPTIMIZER`` payload: the optimizer kind drives its state size."""

    kind: str = "AdamW"

    def describe(self) -> str:
        return self.kind


@dataclasses.dataclass(frozen=True)
class EmbeddingInfo:
    """The embedding link of a spec chain (see ``comfy/lm_protocol.py``)."""

    vocab_size: Optional[int] = None
    d_model: Optional[int] = None
    include_position: bool = True


@dataclasses.dataclass(frozen=True)
class BlockInfo:
    """One transformer-block link of a spec chain."""

    d_model: Optional[int] = None
    num_heads: int = 4
    d_ffn: Optional[int] = None
    activation: str = "relu"
    dropout: float = 0.0


@dataclasses.dataclass(frozen=True)
class SpecVal(EstValue):
    """A ``MODELSPEC`` chain: embedding first, then any number of blocks."""

    embedding: Optional[EmbeddingInfo] = None
    blocks: tuple = ()

    @property
    def is_known(self) -> bool:
        return (
            self.embedding is not None
            and self.embedding.vocab_size is not None
            and self.embedding.d_model is not None
        )

    def describe(self) -> str:
        if not self.is_known:
            return "spec (unknown widths)"
        widths = ", ".join(str(b.d_ffn) for b in self.blocks)
        return (
            f"spec[vocab={self.embedding.vocab_size}, "
            f"d_model={self.embedding.d_model}, blocks={len(self.blocks)}"
            + (f", ffn=[{widths}]" if widths else "")
            + "]"
        )


@dataclasses.dataclass(frozen=True)
class ModelVal(EstValue):
    """An ``NNMODEL`` payload: the spec chain is all the estimator needs."""

    spec: Optional[SpecVal] = None

    def describe(self) -> str:
        return self.spec.describe() if self.spec else "model (unknown structure)"
