"""HTTP routes for the ComfyDL Profiling panel (reform: Profiling Tools M1/M2).

POST endpoints, pure JSON:

* ``/comfydl/profiling/estimate``   - body ``{"prompt": <graphToPrompt output>,
  "budget": {...} (optional), "assumptions": {...} (optional),
  "titles": {...} (optional)}`` -> the full estimation report (M1 memory
  ledger + M2 compute ledger);
* ``/comfydl/profiling/postmortem`` - body ``{"message": <exception text>,
  "node_id", "node_type", "prompt"/"budget"/"assumptions" (optional)}`` ->
  the OOM post-mortem (attribution + suggested batch size).

GET endpoints (M2 watchdog):

* ``/comfydl/profiling/watchdog/status`` - the live sample of the execution
  monitor (current node, process CPU/RSS, system CPU);
* ``/comfydl/profiling/watchdog/log?limit=50`` - the most recent burst
  events from the persistent JSONL ledger.

The frontend asset side (``profiler.js`` / ``profiler.css``) is served from
the repository-owned ``app/profiling_assets/`` directory by a static route
mounted in ``server.py``; the loader that pulls it into the page is injected
by ``app/frontend_patch.py``. Nothing here touches the frontend package.
"""

from __future__ import annotations

import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from aiohttp import web

from comfy.profiling import analyse_oom, estimate_workflow
from comfy.profiling import proflog
from comfy.profiling.engine import DeviceBudget, budget_from_environment
from comfy.profiling.runmeter import set_persistence_hook

#: The panel ships a .ttf font subset under app/profiling_assets/, served by
#: the static route. Python 3.14 maps .ttf to font/ttf out of the box, 3.12
#: does not - and aiohttp >= 3.14 answers through its own private MimeTypes
#: instance (aiohttp.web_fileresponse.CONTENT_TYPES), not the global module.
#: Both registrations together keep the MIME right on every interpreter /
#: aiohttp combination we ship with; caught by T9h on the F drive (3.12).
mimetypes.add_type("font/ttf", ".ttf")
try:
    from aiohttp import web_fileresponse

    web_fileresponse.CONTENT_TYPES.add_type("font/ttf", ".ttf")
except (ImportError, AttributeError):  # older aiohttp: global map is enough
    pass

routes = web.RouteTableDef()

#: The M2 execution watchdog, injected by server.py after construction.
#: Kept as a module global because the route table is declared before the
#: server exists (the /api twin loop in server.py re-declares these).
_WATCHDOG = None

#: The most recent estimation report, so the watchdog can attach a
#: data-scale hint to a burst event (the engine <-> monitor handshake).
_LAST_REPORT: Dict[str, Any] = {}


def set_watchdog(watchdog) -> None:
    """Give the routes the server's watchdog instance."""
    global _WATCHDOG
    _WATCHDOG = watchdog


def scale_hint_for(node_id: str) -> Optional[dict]:
    """What the M1/M2 estimator last said about ``node_id`` (or None)."""
    for node in _LAST_REPORT.get("nodes", []):
        if node.get("id") == node_id:
            return {
                "class_type": node.get("class_type"),
                "total_bytes": node.get("total_bytes"),
                "flops_total": node.get("flops_total"),
                "basis": node.get("basis"),
            }
    return None


def _budget_from_payload(payload: Dict[str, Any]) -> Optional[DeviceBudget]:
    """The caller-supplied budget, else the live primary-device budget."""
    data = payload.get("budget")
    if isinstance(data, dict) and int(data.get("total_bytes") or 0) > 0:
        return DeviceBudget(
            name=str(data.get("name") or "device"),
            total_bytes=int(data["total_bytes"]),
            free_bytes=int(data.get("free_bytes") or data["total_bytes"]),
        )
    return budget_from_environment()


@routes.post("/comfydl/profiling/estimate")
async def estimate(request: web.Request) -> web.Response:
    started = time.perf_counter()
    try:
        payload = await request.json()
    except Exception:
        proflog.log("low", "estimate: 400 invalid JSON body", request=request)
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        proflog.log("low", "estimate: 400 body must be a JSON object", request=request)
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    report = estimate_workflow(
        payload.get("prompt") if isinstance(payload.get("prompt"), dict) else {},
        _budget_from_payload(payload),
        payload.get("assumptions") if isinstance(payload.get("assumptions"), dict) else None,
        payload.get("titles") if isinstance(payload.get("titles"), dict) else None,
    )
    global _LAST_REPORT
    _LAST_REPORT = report
    nodes = report.get("nodes") or []
    report = _merge_measured(report)
    nodes = report.get("nodes") or []
    proflog.log(
        "low", "estimate: %d nodes -> verdict=%s peak=%s in %.0fms",
        len(nodes), report.get("verdict"),
        report.get("peak_bytes"), (time.perf_counter() - started) * 1000.0,
        request=request,
    )
    return web.json_response(report)


#: P0: the probe executes REAL node code with REAL tensors, so it runs on a
#: dedicated single-worker pool instead of the shared default executor - a
#: runaway analysis can no longer starve other profiling routes, and the
#: "cdl-probe" thread name keeps runaway stacks identifiable in dumps. A
#: thread cannot be killed mid-run; the hard timeout below only *abandons*
#: the future (the worker finishes in the background and its result is
#: discarded) - documented in docs/profiling-m3-opgraph.md, Safety model.
_PROBE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cdl-probe")

#: The header the frontend attaches ONLY when the user explicitly enabled the
#: dangerous probe (Settings -> ComfyDL -> Profiling). Without it the route
#: never executes a single node.
DANGEROUS_HEADER = "X-CDL-Profiling-Dangerous"


@routes.post("/comfydl/profiling/opgraph")
async def opgraph(request: web.Request) -> web.Response:
    """The equivalent computation graph endpoint (M3 + Profiling v2 P1).

    Two paths behind one button (manual Analyze, never auto-run):

    * **default (P1)**: ``opgraph.assemble_workflow`` - a zero-execution merge
      of the static rule census and the M2 shape-aware formula ledger.  Safe
      for any input by construction.
    * **dangerous (P0, opt-in)**: the real-execution probe, still gated behind
      the ``X-CDL-Profiling-Dangerous`` header with input guardrails and the
      hard deadline.  Kept for sampling review and cross-validation.
    """
    try:
        payload = await request.json()
    except Exception:
        proflog.log("low", "opgraph: 400 invalid JSON body", request=request)
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        proflog.log("low", "opgraph: 400 body must be a JSON object", request=request)
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    prompt = payload.get("prompt") if isinstance(payload.get("prompt"), dict) else None

    dangerous = request.headers.get(DANGEROUS_HEADER, "") == "1"
    started = time.perf_counter()

    import asyncio

    from comfy.profiling import opgraph

    if not dangerous:
        proflog.log(
            "low", "opgraph: assembling equivalent graph, zero execution (%d nodes)",
            len(prompt) if isinstance(prompt, dict) else 0,
            request=request,
        )
        report = opgraph.assemble_workflow(prompt)
    else:
        proflog.log(
            "low", "opgraph analyze: start (%d graph nodes, DANGEROUS mode)",
            len(prompt) if isinstance(prompt, dict) else 0,
            request=request,
        )
        deadline = time.monotonic() + opgraph.PROBE_DEADLINE_SECONDS
        loop = asyncio.get_running_loop()
        try:
            report = await asyncio.wait_for(
                loop.run_in_executor(
                    _PROBE_POOL, opgraph.analyse_workflow, prompt, None, deadline),
                timeout=opgraph.PROBE_DEADLINE_SECONDS + 5.0,
            )
        except asyncio.TimeoutError:
            # The worker thread cannot be killed: it finishes in the background
            # and its (discarded) result never reaches the cache. The route
            # answers immediately instead of wedging the request forever.
            proflog.log(
                "low", "opgraph analyze: gave up after %.0fs (worker may still finish in background)",
                opgraph.PROBE_DEADLINE_SECONDS + 5.0, request=request,
            )
            return web.json_response({
                "version": 1,
                "mode": "timeout",
                "totals": {"nodes": 0, "probed": 0, "fallback": 0,
                           "flops_formula": 0, "flops_probed": 0, "ratio": None},
                "nodes": [],
                "error": (
                    f"probe timeout after {int(opgraph.PROBE_DEADLINE_SECONDS)}s - "
                    "the analysis was abandoned; results are discarded"
                ),
                "disclaimer_key": "opgraphDisclaimer",
            })
        except Exception as exc:
            proflog.log("low", "opgraph analyze: worker crashed: %r", exc, request=request)
            raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    totals = report.get("totals") or {}
    verb = "assembled" if not dangerous else "analyze"
    proflog.log(
        "low", "opgraph %s: done rule/probed=%s/%s fallback=%s in %.0fms",
        verb, totals.get("probed"), totals.get("nodes"), totals.get("fallback"), elapsed_ms,
        request=request,
    )
    for row in report.get("nodes") or []:
        proflog.log(
            "medium",
            "opgraph: node %s %s -> %s ops=%s flops=%s probe=%sms%s",
            row.get("id"), row.get("class_type"), row.get("status"),
            row.get("ops_total"), row.get("total_flops"), row.get("probe_ms"),
            f" error={row.get('error')}" if row.get("error") else "",
            request=request,
        )
    return web.json_response(report)


@routes.post("/comfydl/profiling/postmortem")
async def postmortem(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except Exception:
        proflog.log("low", "postmortem: 400 invalid JSON body", request=request)
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        proflog.log("low", "postmortem: 400 body must be a JSON object", request=request)
        return web.json_response({"error": "body must be a JSON object"}, status=400)
    result = analyse_oom(payload, _budget_from_payload(payload))
    attributed = result.get("attributed") or {}
    suggestion = result.get("suggestion") or {}
    proflog.log(
        "low",
        "postmortem: node=%s type=%s -> attributed=%s(%s) suggested_batch=%s",
        payload.get("node_id"), payload.get("node_type"),
        attributed.get("label"), attributed.get("bytes"), suggestion.get("batch_size"),
        request=request,
    )
    return web.json_response(result)


@routes.get("/comfydl/profiling/watchdog/status")
async def watchdog_status(request: web.Request) -> web.Response:
    if _WATCHDOG is None:
        proflog.log("high", "watchdog status: unavailable (no watchdog)", request)
        return web.json_response({"available": False, "running": False}, status=503)
    snapshot = _WATCHDOG.snapshot()
    proflog.log(
        "high", "watchdog poll: running=%s node=%s",
        snapshot.get("running"), snapshot.get("node_id"),
        request=request,
    )
    return web.json_response(snapshot)


@routes.get("/comfydl/profiling/watchdog/log")
async def watchdog_log(request: web.Request) -> web.Response:
    if _WATCHDOG is None:
        return web.json_response({"events": [], "available": False}, status=503)
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        limit = 50
    events = _WATCHDOG.recent(limit=limit)
    proflog.log("high", "watchdog log: %d events (limit=%d)", len(events), limit, request=request)
    return web.json_response({"events": events})


# --------------------------------------------------------------------------
# P2: measured-FLOPs persistence + history + estimate calibration.
#
# The executor (execution.py) attributes dispatched ATen ops to nodes via
# RunMeter and hands the finished prompt's snapshot to persist_run(); the
# hook registered below routes it into the user database.  Courtesy rule:
# nothing here may break a run - every failure degrades to "no data".


def _persist_measurement(prompt_id: str, prompt: dict, snapshot: dict) -> None:
    """App-layer callback for ``runmeter.set_persistence_hook``."""
    import uuid

    from app.database import db as app_db
    from app.database.models import ProfilingNodeStat, ProfilingRun

    if not app_db.can_create_session():
        return
    if not (snapshot.get("nodes") or {}):
        return  # nothing measured (e.g. a run with no dispatchable ops)
    run_id = str(uuid.uuid4())
    nodes_snap = snapshot.get("nodes") or {}
    with app_db.create_session() as session:
        session.add(ProfilingRun(
            id=run_id,
            prompt_id=str(prompt_id)[:36],
            mode=str(snapshot.get("mode") or "count"),
            total_flops=int(snapshot.get("total_flops") or 0),
            total_ops=int(snapshot.get("total_ops") or 0),
            node_count=len(nodes_snap),
            unattributed_flops=int(
                (snapshot.get("unattributed") or {}).get("flops") or 0),
            sampled=bool(snapshot.get("sampled")),
            suppressed_errors=int(snapshot.get("suppressed_errors") or 0),
        ))
        for node_id, data in nodes_snap.items():
            node = prompt.get(str(node_id)) or {}
            session.add(ProfilingNodeStat(
                id=str(uuid.uuid4()),
                run_id=run_id,
                node_id=str(node_id)[:64],
                class_type=str(node.get("class_type") or "")[:128],
                flops=int(data.get("flops") or 0),
                op_count=int(data.get("op_count") or 0),
                ops=data.get("ops") if snapshot.get("mode") == "census" else None,
                demoted=bool(data.get("demoted")),
            ))
        session.commit()
    proflog.log(
        "low", "history: run %s stored (%d nodes, %s flops)",
        run_id[:8], len(nodes_snap), snapshot.get("total_flops"),
    )


set_persistence_hook(_persist_measurement)


def _measured_average_by_class_type() -> Dict[str, Dict[str, Any]]:
    """Per-class_type measured-FLOPs averages over recorded runs.

    Empty dict when the database is unavailable - the estimate merge then
    simply adds no measured fields.
    """
    from sqlalchemy import func, select

    from app.database import db as app_db
    from app.database.models import ProfilingNodeStat

    if not app_db.can_create_session():
        return {}
    try:
        with app_db.create_session() as session:
            rows = session.execute(
                select(
                    ProfilingNodeStat.class_type,
                    func.avg(ProfilingNodeStat.flops),
                    func.count(ProfilingNodeStat.id),
                )
                .group_by(ProfilingNodeStat.class_type)
            ).all()
        return {
            str(row[0]): {"measured_avg": float(row[1] or 0), "samples": int(row[2] or 0)}
            for row in rows
        }
    except Exception as exc:  # noqa: BLE001 - calibration is a courtesy
        proflog.log("low", "history: measured averages unavailable: %r", exc)
        return {}


def _merge_measured(report: dict) -> dict:
    """Attach measured_flops / measured_ratio to nodes with history."""
    measured = _measured_average_by_class_type()
    if not measured:
        return report
    for node in report.get("nodes") or []:
        entry = measured.get(str(node.get("class_type")))
        if not entry or not entry["samples"]:
            continue
        node["measured_flops"] = entry["measured_avg"]
        node["measured_samples"] = entry["samples"]
        formula = node.get("flops_total")
        if isinstance(formula, (int, float)) and formula > 0 and entry["measured_avg"] > 0:
            node["measured_ratio"] = round(entry["measured_avg"] / formula, 4)
    return report


@routes.get("/comfydl/profiling/history")
async def history(request: web.Request) -> web.Response:
    """Recent measured runs (newest first) with their per-node stats."""
    try:
        limit = max(1, min(100, int(request.query.get("limit", "10"))))
    except ValueError:
        limit = 10
    from sqlalchemy import select

    from app.database import db as app_db
    from app.database.models import ProfilingNodeStat, ProfilingRun

    if not app_db.can_create_session():
        return web.json_response({"runs": [], "available": False})
    try:
        with app_db.create_session() as session:
            run_rows = session.execute(
                select(ProfilingRun)
                .order_by(ProfilingRun.started_at.desc(), ProfilingRun.id)
                .limit(limit)
            ).scalars().all()
            runs = []
            for run in run_rows:
                stats = session.execute(
                    select(ProfilingNodeStat)
                    .where(ProfilingNodeStat.run_id == run.id)
                    .order_by(ProfilingNodeStat.node_id)
                ).scalars().all()
                runs.append({
                    "id": run.id,
                    "prompt_id": run.prompt_id,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "mode": run.mode,
                    "total_flops": run.total_flops,
                    "total_ops": run.total_ops,
                    "node_count": run.node_count,
                    "unattributed_flops": run.unattributed_flops,
                    "nodes": [{
                        "node_id": s.node_id,
                        "class_type": s.class_type,
                        "flops": s.flops,
                        "op_count": s.op_count,
                        "demoted": s.demoted,
                    } for s in stats],
                })
    except Exception as exc:  # noqa: BLE001 - history is a courtesy
        proflog.log("low", "history: query failed: %r", exc)
        return web.json_response({"runs": [], "available": False}, status=200)
    proflog.log("low", "history: %d run(s) served", len(runs), request=request)
    return web.json_response({"runs": runs, "available": True})
