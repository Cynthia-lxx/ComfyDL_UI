"""Live preview helpers for the long-running ComfyDL nodes (reform step 11).

ComfyUI renders two things under the *currently executing node*: a progress
bar and, when one is attached, a preview image. Both travel the same official
side channel - ``comfy.utils.ProgressBar.update_absolute(value, total,
preview)`` - whose global hook (installed by ``main.py``) resolves the node
from the execution context, pushes the numbers as the ``progress`` event and
the image as a binary event. A node therefore needs neither its own id nor
any server machinery to show a live picture; it just passes the third
argument. The accepted preview format is the same triple KSampler's latent
preview uses (``latent_preview.py`` / ``server.py:send_image_with_metadata``):

    ("JPEG", PIL.Image.Image, max_size)

This module turns the two things ComfyDL's trainers actually want to watch
into such images:

* :class:`LossCurvePreviewer` - a live loss curve for ``Training Loop`` and
  ``Language Model Train``: one call per optimizer step, internally
  rate-limited so a fast loop does not spend its time drawing, plus a
  guaranteed final frame. It prefers matplotlib (Agg backend, imported
  lazily) and silently falls back to a hand-drawn PIL polyline when
  matplotlib is unavailable, so a visualisation gap can never break a
  training run. Every failure path returns ``None`` (no preview) instead of
  raising: a preview is a courtesy, not a contract.
* :func:`render_text_snapshot` - the generated-so-far text of ``Language
  Model Generate``, wrapped onto a small card so a long generation can be
  read while it happens.

Both helpers are pure: no ComfyUI imports, no global state, safe to call
from any thread (matplotlib figures are created and closed locally).
"""

from __future__ import annotations

import io
import time
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# The preview triple format the hook / server understands.
PREVIEW_FORMAT = "JPEG"
# Preview images are re-scaled by the server (MAX_PREVIEW_RESOLUTION); this
# size only bounds our own rendering cost.
PREVIEW_MAX_SIZE = 512

# A preview-bearing update bypasses the ProgressBar's 100 ms throttle (it is
# always sent immediately), so the drawing itself has to be rate-limited.
MIN_RENDER_INTERVAL = 0.5


def _to_image_bytes(image: Image.Image) -> tuple[str, Image.Image, int] | None:
    """Wrap a PIL image into the ``("JPEG", image, max_size)`` preview triple.

    The server re-encodes whatever it receives, so the image is handed over
    as RGB; returning ``None`` on any failure keeps the caller safe.
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return (PREVIEW_FORMAT, image, PREVIEW_MAX_SIZE)
    except Exception:  # noqa: BLE001 - a preview must never raise
        return None


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A best-effort font; the bitmap default keeps text readable-ish."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: load_default takes no size
        return ImageFont.load_default()


class LossCurvePreviewer:
    """Live loss-curve preview for the training-loop nodes.

    What: collects the per-step loss values of a training run and renders
          the convergence curve as a preview image whenever enough time has
          passed since the last render (``min_interval`` seconds, so a
          thousands-of-steps loop does not draw thousands of pictures).
          The final step always renders, so the last pushed frame is the
          complete curve. Rendering is matplotlib (Agg) when available and
          a PIL fallback otherwise; both draw the same information - the
          raw per-step loss, a smoothed (moving-average) overlay and the
          current best value.
    In:   record(loss) - called once per optimizer step with the detached
          scalar loss of that step; returns the preview triple *or* ``None``
          (throttled or failed) for that step.
          render() - draw the curve from the values recorded so far; the
          return value is the preview triple or ``None``.
          reset() - drop the recorded values and timers (a previewer is
          single-run by design).
    Out:  the preview triples are consumed by
          ``comfy.utils.ProgressBar.update_absolute``'s third argument.
    """

    def __init__(
        self,
        title: str = "loss",
        min_interval: float = MIN_RENDER_INTERVAL,
        max_points: int = 4096,
    ) -> None:
        self.title = str(title)
        self.min_interval = max(0.0, float(min_interval))
        self.max_points = int(max_points)
        self._values: list[float] = []
        self._last_render = 0.0

    # -- data collection ----------------------------------------------------

    def reset(self) -> None:
        """Forget everything recorded so far (start of a new run)."""
        self._values.clear()
        self._last_render = 0.0

    def record(
        self, loss: float, force: bool = False
    ) -> tuple[str, Image.Image, int] | None:
        """Record one step's loss and maybe render a preview frame.

        Returns the preview triple when this step should push a frame,
        otherwise ``None``. ``force=True`` (the caller's last step) always
        renders, so the last pushed frame is the complete curve; a throttled
        step returns ``None``, and the caller still reports through plain
        ``update`` - one progress call per step either way.
        """
        try:
            self._values.append(float(loss))
            if len(self._values) > self.max_points:
                del self._values[: len(self._values) - self.max_points]
        except (TypeError, ValueError):
            return None
        if not force:
            now = time.perf_counter()
            if now - self._last_render < self.min_interval:
                return None
        self._last_render = time.perf_counter()
        return self.render()

    # -- rendering ----------------------------------------------------------

    def render(self) -> tuple[str, Image.Image, int] | None:
        """Draw the curve of the values recorded so far."""
        if not self._values:
            return None
        image = None
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            image = self._render_matplotlib()
        except Exception:  # noqa: BLE001 - fall back to PIL on any failure
            image = None
        if image is None:
            try:
                image = self._render_pil()
            except Exception:  # noqa: BLE001 - a preview must never raise
                return None
        return _to_image_bytes(image)

    # matplotlib path --------------------------------------------------------

    def _render_matplotlib(self):
        import matplotlib.figure

        values = self._values
        figure = matplotlib.figure.Figure(figsize=(4.0, 3.0), dpi=100)
        axis = figure.add_subplot(1, 1, 1)
        steps = range(1, len(values) + 1)
        axis.plot(steps, values, color="#1f77b4", linewidth=1.2, label="loss")
        smoothed = _moving_average(values, 8)
        if len(values) >= 8:
            axis.plot(steps, smoothed, color="#ff7f0e", linewidth=1.6, label="smoothed")
        axis.set_title(f"{self.title}  (step {len(values)}, best {min(values):.4g})")
        axis.set_xlabel("step")
        axis.set_ylabel("loss")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    # PIL fallback path ------------------------------------------------------

    def _render_pil(self) -> Image.Image:
        width, height = 400, 300
        margin_left, margin_right = 44, 10
        margin_top, margin_bottom = 30, 10
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        values = self._values
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo + 1.0

        def _x(index: int) -> float:
            span = max(1, len(values) - 1)
            return margin_left + (width - margin_left - margin_right) * index / span

        def _y(value: float) -> float:
            return margin_top + (height - margin_top - margin_bottom) * (
                1.0 - (value - lo) / (hi - lo)
            )

        # Frame and y labels.
        draw.rectangle(
            [margin_left - 1, margin_top - 1, width - margin_right, height - margin_bottom],
            outline="#888888",
        )
        font = _load_font(10)
        for fraction in (0.0, 0.5, 1.0):
            y = margin_top + (height - margin_top - margin_bottom) * fraction
            value = hi - (hi - lo) * fraction
            draw.text((2, y - 5), f"{value:.3g}", fill="#333333", font=font)
            draw.line(
                [margin_left, y, width - margin_right, y], fill="#dddddd", width=1
            )
        # Raw curve.
        points = [(_x(i), _y(v)) for i, v in enumerate(values)]
        if len(points) >= 2:
            draw.line(points, fill="#1f77b4", width=2)
        # Smoothed overlay.
        smoothed = _moving_average(values, 8)
        if len(values) >= 8:
            smooth_points = [(_x(i), _y(v)) for i, v in enumerate(smoothed)]
            draw.line(smooth_points, fill="#ff7f0e", width=2)
        draw.text(
            (margin_left, 8),
            f"{self.title}  step {len(values)}  best {min(values):.4g}",
            fill="#111111",
            font=font,
        )
        return image


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    """Trailing moving average aligned with the input (right-anchored)."""
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        count = min(index + 1, window)
        result.append(running / count)
    return result


def render_text_snapshot(
    text: str,
    title: str = "generated",
    max_chars: int = 600,
) -> tuple[str, Image.Image, int] | None:
    """Render the generated-so-far text as a preview card.

    What: the text-preview counterpart of the loss curve - a small white
          card holding the text generated so far (tail-truncated to
          ``max_chars``), so a long autoregressive run can be read while it
          happens. Words are wrapped at ~46 characters per line.
    In:   text - the accumulated text (prefix + generated tokens so far).
          title - card header, e.g. ``"generated 12/64"``.
          max_chars - keep at most this many trailing characters.
    Out:  the preview triple for ``update_absolute``, or ``None`` on any
          failure (a preview must never break the generation loop).
    """
    try:
        body = str(text)[-int(max_chars):] if text else ""
        wrapped = _wrap(body, width=46)
        line_count = max(1, len(wrapped))
        line_height = 16
        width, height = 420, max(90, 34 + line_count * line_height + 12)
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        font = _load_font(12)
        draw.text((8, 8), str(title), fill="#1f77b4", font=font)
        for index, line in enumerate(wrapped):
            draw.text((8, 30 + index * line_height), line, fill="#111111", font=font)
        return _to_image_bytes(image)
    except Exception:  # noqa: BLE001 - a preview must never raise
        return None


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap; hard-splits tokens longer than ``width``."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines if lines else [""]
