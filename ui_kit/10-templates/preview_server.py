#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Preview server for the UI reference kit. Python 3.9+, standard library only.
#
# Usage:
#   python preview_server.py --port 8190
#   python preview_server.py --port 8190 --assemble      # run assemble.py first
#   python preview_server.py --root <UIReference> --dist <dist dir>
#
# Routes:
#   /                 -> dist/preview.html   (full preview site)
#   /classic          -> dist/index.html    (original single-page demo)
#   /<path>           -> static file from dist/
#   /api/health       -> server / dist status
#   /api/manifest /inventory /motion /themes /settings /i18n/<lang> /audit
#   /api/files        -> generated dist file list
#   /api/validate     -> run assemble.py validation and return its summary
#
# All responses are sent with Cache-Control: no-store so a rebuild is visible
# on a plain refresh (no hard refresh needed).

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PORT = 8190
HOST = "127.0.0.1"
QUIET = False

ENDPOINTS = [
    "/api/health", "/api/manifest", "/api/inventory", "/api/expansion",
    "/api/layers", "/api/motion", "/api/themes", "/api/settings",
    "/api/i18n/<lang>", "/api/audit", "/api/files", "/api/validate",
]

MIME = {
    "": "application/octet-stream",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".webp": "image/webp",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dist_files() -> list:
    if not DIST.is_dir():
        return []
    return sorted(str(p.relative_to(DIST)).replace("\\", "/") for p in DIST.rglob("*") if p.is_file())


def load_assemble_module():
    path = ROOT / "10-templates" / "assemble.py"
    spec = importlib.util.spec_from_file_location("ui_kit_assemble", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_api(name: str):
    """Return a JSON-serialisable payload for an /api/* name, or None if unknown."""
    if name == "health":
        return {
            "ok": True,
            "host": HOST,
            "port": PORT,
            "root": str(ROOT),
            "dist": str(DIST),
            "files": len(dist_files()),
            "python": platform.python_version(),
            "endpoints": ENDPOINTS,
            "routes": {"/": "preview.html", "/classic": "index.html"},
        }
    if name == "manifest":
        return load_json(ROOT / "00-overview/ui-manifest.json")
    if name == "inventory":
        return load_json(ROOT / "05-motion/effect-inventory.json")
    if name == "expansion":
        return load_json(ROOT / "05-motion/expansion-inventory.json")
    if name == "layers":
        return load_json(ROOT / "05-motion/layered-view.json")
    if name == "motion":
        return load_json(ROOT / "05-motion/motion.json")
    if name == "themes":
        return load_json(ROOT / "03-colour-typography/themes/index.json")
    if name == "settings":
        return load_json(ROOT / "00-overview/settings.json")
    if name == "audit":
        return load_json(ROOT / "00-overview/licence-audit.json")
    if name == "files":
        return {"count": len(dist_files()), "files": dist_files()}
    if name == "validate":
        module = load_assemble_module()
        summary = module.assemble(ROOT, DIST, "blue", "en", True)
        return {"ok": summary.get("validation") == "ok", "summary": summary}
    if name.startswith("i18n/"):
        language = name.split("/", 1)[1]
        if not language or any(ch in language for ch in "/\\.."):
            return {"error": "invalid language code"}
        path = ROOT / "00-overview/i18n" / f"{language}.json"
        if not path.is_file():
            return {"error": f"language not found: {language}"}
        return load_json(path)
    return None


class PreviewHandler(SimpleHTTPRequestHandler):
    server_version = "UIReferencePreview/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            return self.handle_api(path[len("/api/"):].strip("/"))
        if path in ("", "/"):
            self.path = "/preview.html"
        elif path in ("/classic", "/classic/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            return self.handle_api(path[len("/api/"):].strip("/"), head=True)
        if path in ("", "/"):
            self.path = "/preview.html"
        elif path in ("/classic", "/classic/"):
            self.path = "/index.html"
        return super().do_HEAD()

    def handle_api(self, name: str, head: bool = False):
        try:
            payload = build_api(name)
        except Exception as exc:  # noqa: BLE001 - surface any server-side failure as JSON
            return self.send_json({"error": str(exc), "endpoint": name}, 500)
        if payload is None:
            return self.send_json({"error": "not found", "endpoint": name}, 404)
        return self.send_json(payload, 200, head=head)

    def send_json(self, payload, status: int = 200, head: bool = False):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def guess_type(self, path):
        return MIME.get(Path(path).suffix.lower(), super().guess_type(path))

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt, *args):
        if QUIET:
            return
        try:
            message = fmt % args
        except Exception:  # noqa: BLE001
            message = str(args)
        if message.startswith('"GET /api/') or message.startswith('"GET /preview') or " 404 " in message or " 500 " in message:
            sys.stderr.write(f"[preview] {message}\n")


def main(argv=None) -> int:
    global ROOT, DIST, PORT, HOST, QUIET

    parser = argparse.ArgumentParser(description="Preview server for the UI reference kit (Python 3).")
    parser.add_argument("--root", type=Path, default=ROOT, help="UIReference root directory")
    parser.add_argument("--dist", type=Path, default=None, help="dist directory (default: <root>/dist)")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8190, help="port (default: 8190)")
    parser.add_argument("--assemble", action="store_true", help="run assemble.py before serving")
    parser.add_argument("--quiet", action="store_true", help="suppress request logs")
    parser.add_argument("--open", action="store_true", help="open the preview in the default browser")
    args = parser.parse_args(argv)

    ROOT = args.root.resolve()
    DIST = (args.dist or (ROOT / "dist")).resolve()
    PORT = args.port
    HOST = args.host
    QUIET = args.quiet

    if not DIST.is_dir():
        print(f"[preview] ERROR: dist directory not found: {DIST}", file=sys.stderr)
        print("[preview] run: python 10-templates/assemble.py", file=sys.stderr)
        return 1

    if args.assemble:
        try:
            module = load_assemble_module()
            result = module.assemble(ROOT, DIST, "blue", "en", False)
            print(f"[preview] assembled {len(result.get('generatedFiles', []))} files into {DIST}")
        except Exception as exc:  # noqa: BLE001
            print(f"[preview] ERROR: assemble failed: {exc}", file=sys.stderr)
            return 1

    if not (DIST / "preview.html").is_file():
        print(f"[preview] WARNING: {DIST / 'preview.html'} missing; run assemble.py with the preview support", file=sys.stderr)

    handler = partial(PreviewHandler)
    server = ThreadingHTTPServer((HOST, PORT), handler)
    server.daemon_threads = True
    url = f"http://{HOST}:{PORT}/"
    print("=" * 68)
    print("  UI Reference Kit · preview server")
    print(f"  preview : {url}")
    print(f"  classic : {url}classic")
    print(f"  API     : {url}api/health  (also /api/validate, /api/files, …)")
    print(f"  root    : {ROOT}")
    print(f"  dist    : {DIST}  ({len(dist_files())} files)")
    print("  stop    : Ctrl+C")
    print("=" * 68, flush=True)

    if args.open:
        import webbrowser
        webbrowser.open(url)

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[preview] shutting down…")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
