"""The assumption layer of ``comfy.profiling``.

Static analysis cannot know everything: a token stream may arrive from a
node the estimators do not cover, a checkpoint's structure is not declared in
the graph. Instead of guessing silently, an estimator that hits such an
unknown asks the :class:`AssumptionSet` for a *tiered default*, records that
it did, and the report shows "based on assumption N" prominently.

Values arriving from the user (widget texts, linked vocab sizes) are always
exact and never go through this layer - assumptions only fill genuine holes.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

#: Default tiers for the quantities the engine may have to assume.
#: ``samples``: dataset size when the input stream is not statically known.
#: ``sequence_length``: stream/window length under the same conditions.
#: ``latent_channels`` / ``vae_scale``: the SD AutoencoderKL defaults kept in
#: ``comfy/sd.py`` (VAE.__init__); other architectures override them, which is
#: why they live here as overridable assumptions rather than constants.
#: ``clip_tokens`` / ``clip_hidden``: the conditioning size of a CLIP text
#: encoder - this build keeps no implementation, so both are assumptions.
DEFAULT_ASSUMPTIONS: Dict[str, int] = {
    "samples": 10000,
    "sequence_length": 64,
    "latent_channels": 4,
    "vae_scale": 8,
    "clip_tokens": 77,
    "clip_hidden": 768,
}


@dataclasses.dataclass
class AssumptionSet:
    """User overrides layered over the default tiers.

    ``used`` records every lookup that actually happened, so the report can
    state exactly which numbers were assumed rather than measured.
    """

    overrides: Dict[str, int] = dataclasses.field(default_factory=dict)
    used: Dict[str, Tuple[int, str]] = dataclasses.field(default_factory=dict)

    def get(self, key: str) -> Tuple[int, str]:
        """Return ``(value, source)`` for ``key`` and record the lookup.

        ``source`` is ``"override"`` when the user set the value in the
        Profiling panel and ``"default"`` for the built-in tier.
        """
        if key in self.overrides:
            value, source = int(self.overrides[key]), "override"
        else:
            value, source = int(DEFAULT_ASSUMPTIONS.get(key, 0)), "default"
        self.used[key] = (value, source)
        return value, source

    def used_entries(self) -> list:
        """The recorded lookups, report-ready."""
        return [
            {"key": key, "value": value, "source": source}
            for key, (value, source) in self.used.items()
        ]
