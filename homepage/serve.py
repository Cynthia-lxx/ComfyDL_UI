"""Static file server for the ComfyFront_sep homepage (used by 启动主页.bat).

Standard http.server, plus two things the demo needs:
  * ``Cache-Control: no-store`` on every response — the compiled frontend
    assets are patched in place by bake_patches.py, so a browser that cached
    an older chunk would silently run stale code.
  * a request log line on the server console (the "backend log" for
    startup tracing).

Run: python serve.py [port]   (default 8188, bound to 127.0.0.1)
"""

import http.server
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write("[serve] %s - %s\n" % (self.address_string(), fmt % args))
        sys.stdout.flush()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), NoCacheHandler)
    url = f"http://127.0.0.1:{port}/"
    print(f"[serve] serving {ROOT} at {url}  (Ctrl+C to stop)")
    # Open the browser only after the socket is bound — the old launcher
    # opened it before the server started, which raced the first page load.
    threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")


if __name__ == "__main__":
    main()
