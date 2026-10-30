# CDL App Panel — App-mode companion for deep learning work

*2026-10-30. Companion panel for the official beta App mode (internal codename
"linear mode"), rebuilt around what deep learning actually produces: tensors,
models, text and files — not gallery images.*

## Native dock (batch 5-7, full-bleed zoned CDLApp UI)

The App screen becomes ONE CDLApp UI with PLANNED ZONES — no dead empty
space, no vertical cramming. In App mode the official input panel
(`.arrange-panel`, the real Vue widgets the frontend builds for app-mode
workflows) is **re-positioned by CSS to cover the full remaining width**
(`position:fixed` from the side tool bar to the right edge). Inside it, the
official body column becomes a two-column grid:

- **Left zone (narrow, ~340px)** — the official params widgets + Run button.
- **Right zone (rest)** — the CDL dashboard (`#cdl-app-extras`) as a
  named-area grid: **previews** as the hero tile (top left), the long
  **outputs** list as its own tall column (top right), **progress** under
  previews, **errors** as a full-width strip at the bottom. Every tile
  scrolls inside its own box — the screen never grows.

The CDL sections are appended **inside** the official panel as a plain
sibling, so they inherit its fonts, colors and padding and look native by
construction. The two zone boundaries are **draggable**: a 9px gutter track
with a handle separates the params column from the dashboard, and the preview
hero tile from the outputs column (divider line lights up teal on
hover/drag; sizes persist in `localStorage` under
`ComfyDL.AppPanel.ParamsW` / `ComfyDL.AppPanel.PrevW`). Image artifacts render as a clickable **preview thumbnail
grid** (official `/view` endpoint, newest first, capped at 12; click opens
the full image). The dock is scoped under `body.cdlapp-active` (set only
while our extras are mounted) and releases cleanly when App mode exits —
graph mode is untouched; below 1100px the zones collapse to a vertical
fallback.

## Why

The official App view is built for image-generation workflows: the output display
expects pretty previews and (on Comfy Cloud) the `/jobs/{id}/assets` endpoint.
A DL run's interesting results are a CSV of predictions, a saved `.pt` state
dict, a text preview of a dataset, a loss curve. The CDL App Panel makes those
first-class:

- **Outputs** — every run collects (a) files listed from the output directory
  via `GET /comfydl/artifacts` (download buttons, newest first), and (b) text /
  image entries from the `executed` event payload (copy buttons; image entries
  link to `/view`). Text outputs that reference a written file graduate to
  download rows. Low-cost previews come from the existing renderers
  (CdlPlot / CdlHeatmaps render real matplotlib PNGs) — no new preview machinery.
- **Progress** — per-node rows driven purely by the official `progress_state` /
  `executing` / `executed` / `execution_cached` WebSocket events: node title,
  percent, running/done/cached/error state, plus a total bar. The panel only
  consumes events; it never emits pbar calls (the one-pbar-per-step contract).
- **Errors ("die understanding")** — `execution_error` becomes a card with the
  node title/type, the exception message and the full traceback, collapsed by
  default, one click to copy everything.
- **Navigation** — *Node graph* executes the official `Comfy.ToggleLinear`
  command; *Inputs* / *Outputs* jump straight into the official builder at the
  matching step (probed at runtime, never hardcoded store symbols).

The whole App screen is the zoned CDLApp UI — nothing covered, nothing
wasted; graph mode keeps its own full layout when App mode is off.

## Files

| File | Role |
|---|---|
| `app/cdlapp_assets/cdlapp.js` | Panel: detection, sections, previews, nav, settings switch |
| `app/cdlapp_assets/cdlapp.css` | Full-bleed dock layout, official-variable theming, z-index 2100 |
| `app/cdlapp_routes.py` | `GET /comfydl/artifacts[/download]` (path-jailed) |
| `app/frontend_patch.py` | `cdlapp` loader injection + zh bake for the setting |
| `cdl_smoke_tests/test_cdlapp.py` | 71-check suite (see below) |

## Official-evolution containment (ADR-8)

Every dependency on official internals is declared once in `OFFICIAL_SURFACE`
(frontend version, detection anchors, event names, command ids, builder probes,
z-index band). Feature capability is probed at runtime; a drifted surface
**degrades one section** with a console warning — it never blanks the app.

When the frontend pip package is upgraded, `test_cdlapp.py` re-pins every
surface entry against the installed chunks and fails loudly with migration
guidance. Live-verified facts (2026-10-30, agent-browser store enumeration —
an earlier subagent report was wrong on all three):

- The pinia store id is `appMode`; it carries a `viewMode` string
  (`'graph'` / `'app'` / `'builder:*'`) and `enterBuilder()` / `exitBuilder()`
  (no args). There is **no** `isAppMode` flag, **no** `setMode()` method, and
  the coachmark ids / `data-testid` strings never render as DOM attributes.
- Mode flips go through `extensionManager.command.execute('Comfy.ToggleLinear')`.
- The settings gate must read the **pinia** settings store
  (`$pinia._s.get("setting").isReady`); the workspace-store `setting` field
  has no `isReady` and made the old gate time out (30s blank wait).

The artifacts payload mirrors the official cloud asset schema
(`name`/`media_type` fields) so a future official local jobs/assets
implementation can replace `cdlapp_routes.py` without frontend changes.

## Settings

- **`ComfyDL.AppPanel.Enabled`** (default on, Settings → ComfyDL → App) — the
  master switch; off means zero DOM injection. Like every ComfyDL extension,
  registration waits for the settings store to signal ready (the 2026-10-30
  white-screen rule).

## Path jail (artifacts)

Both artifact endpoints resolve the requested name inside the ComfyUI output
directory: absolute paths, `..` escapes and symlinks anywhere along the path
are rejected with 403 (the symlink check runs on unresolved components —
`resolve()` would follow the link). Media types come from an explicit table
(the Windows mimetypes registry maps `.csv` to `application/vnd.ms-excel`).

## Try it

Open **Templates → ComfyDL Examples → Formula Playground: App Mode**
(`formula_playground_app`) — it opens in App mode by default. Run it, watch
the per-node progress, then download the artifacts or edit the formula and
re-run. The template carries `extra.linearMode` / `extra.linearData`
(the official contract) and `isApp: true` (the template-browser badge).
