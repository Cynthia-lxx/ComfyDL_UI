"""torch.compile wrapper node for MODEL (dual-track M-B, SP-3).

A single MODEL->MODEL node that compiles the diffusion model's forward with
``torch.compile``, through the central :mod:`comfy.compile_manager` (one
compilation per module+settings, weakref-cached, never an in-place
``setattr`` - the ComfyUI-Torch-Compile defect).  The compiled wrapper is
attached via ``ModelPatcher.add_object_patch`` (the WaveSpeed pattern), so
unrelated MODEL branches are untouched and the patch dies with the clone.

Known blind spot, documented by design (bigplan ADR-D4): compiled regions
bypass ``TorchDispatchMode``, so the profiling run-meter cannot see
individual ATen calls inside them - attribution degrades to node
granularity, and the profiling opgraph probe should run under
``torch.compiler.disable`` (planned T6-2).  Compilation wall time is booked
into the ``comfydl.compile`` switch (meta-profiling rule).
"""

from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from comfy import compile_manager

CATEGORY = "model/perf"

#: torch.compile modes worth exposing; ``default`` maps to ``None`` (no mode).
_MODES = {
    "default": None,
    "reduce-overhead": "reduce-overhead",
    "max-autotune": "max-autotune",
    "max-autotune-no-cudagraphs": "max-autotune-no-cudagraphs",
}

#: torch.compile backends that make sense here; ``eager``/``aot_eager`` are
#: the debugging pair (trace without fusing).
_BACKENDS = ["inductor", "cudagraphs", "eager", "aot_eager"]


def _require_model(model) -> None:
    if model is None:
        raise RuntimeError(
            "CdlCompileModel needs a real MODEL value - wire a checkpoint or "
            "diffusion-model loader in front of this node.")


class CdlCompileModel(io.ComfyNode):
    """Compile the diffusion model's forward with torch.compile.

    What: clones the incoming MODEL and registers a compiled wrapper for its
          ``diffusion_model`` forward via ``add_object_patch`` - the graph's
          other branches are untouched.  The compilation itself goes through
          ``comfy.compile_manager``: it happens once per (module, settings)
          pair, is weak-cached (re-executing this node with the same model
          and settings is a cache hit), and its wall time is booked into the
          ``comfydl.compile`` switch.  First execution after the patch pays
          the compilation cost (warmup runs it too); later runs are pure
          kernel speed.
    In:   model (MODEL) - the patcher whose ``diffusion_model`` is compiled.
          backend (COMBO, default ``inductor``) - ``inductor`` generates
          fused Triton/CPP kernels; ``cudagraphs`` records CUDA graphs;
          ``eager``/``aot_eager`` trace without fusing (debugging).
          mode (COMBO, default ``default``) - ``reduce-overhead`` enables
          CUDA graphs inside inductor; ``max-autotune`` benchmarks kernel
          variants (slowest compile, often fastest run).
          fullgraph (BOOLEAN) - refuse graph breaks instead of falling back
          piecewise (strict; may fail on models with data-dependent control
          flow).
          dynamic (BOOLEAN) - trace with dynamic shapes (recompiles less
          across resolutions; sometimes slower kernels).
    Out:  MODEL - the clone with the compiled forward attached.
    Caveats: compiled regions are opaque to the profiling run-meter (they
          bypass TorchDispatchMode) - node-level timing still applies.  The
          ``inductor`` backend needs Triton, which stock Windows PyTorch does
          not ship: there the node raises ``TritonMissing`` on first run -
          switch to ``cudagraphs``/``aot_eager`` (or install triton-windows).
          If compilation fails, the error surfaces on the node like any other;
          disable the node (or the ``comfydl.compile`` switch) to bypass.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CdlCompileModel",
            display_name="Compile Model (torch.compile)",
            category=CATEGORY,
            description=(
                "Compile this MODEL branch's diffusion_model forward with "
                "torch.compile (cached once per module+settings; add_object_"
                "patch pattern, no in-place rewrites)."),
            inputs=[
                io.Model.Input("model", tooltip="MODEL whose diffusion_model "
                               "forward gets compiled."),
                io.Combo.Input("backend", options=_BACKENDS, default="inductor",
                               tooltip="torch.compile backend. inductor needs "
                                       "Triton (absent on stock Windows - use "
                                       "cudagraphs/aot_eager there)."),
                io.Combo.Input("mode", options=list(_MODES), default="default",
                               tooltip="Compilation mode (default = no mode)."),
                io.Boolean.Input("fullgraph", default=False,
                                 tooltip="Fail on graph breaks instead of "
                                         "falling back piecewise."),
                io.Boolean.Input("dynamic", default=False,
                                 tooltip="Trace with dynamic shapes (fewer "
                                         "recompiles across resolutions)."),
            ],
            outputs=[io.Model.Output(display_name="MODEL")],
            search_aliases=[
                "compile", "torch.compile", "inductor", "cudagraphs",
                "speedup", "kernel", "fuse",
            ],
        )

    @classmethod
    def execute(cls, model, backend: str = "inductor", mode: str = "default",
                fullgraph: bool = False, dynamic: bool = False) -> io.NodeOutput:
        _require_model(model)
        patcher = model.clone()
        compiled = compile_manager.compile_module(
            patcher.get_model_object("diffusion_model"),
            backend=backend,
            mode=_MODES.get(mode, None),
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        patcher.add_object_patch("diffusion_model", compiled)
        return io.NodeOutput(patcher)


class CompileExtension(ComfyExtension):
    """Registers the ``model/perf`` compile node."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [CdlCompileModel]


async def comfy_entrypoint() -> CompileExtension:
    return CompileExtension()
