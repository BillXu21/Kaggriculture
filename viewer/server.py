"""Serve the repository-local debug-trace viewer without third-party packages."""

from __future__ import annotations

import argparse
import mimetypes
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


VIEWER_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = VIEWER_ROOT.parent / "artifacts" / "debug_traces"
ARTIFACT_URL_PREFIX = "/artifacts/debug_traces/"
ASSET_PATHS = {
    "/viewer/": VIEWER_ROOT / "index.html",
    "/viewer/index.html": VIEWER_ROOT / "index.html",
    "/viewer/viewer.js": VIEWER_ROOT / "viewer.js",
    "/viewer/styles.css": VIEWER_ROOT / "styles.css",
}


class ViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve only the viewer assets and JSON debug-trace artifacts."""

    server_version = "KaggricultureViewer/1.0"

    def _route(self) -> Path | None:
        """Map a URL path to an allowed local file, never the repository root."""
        request_path = unquote(urlsplit(self.path).path)
        asset = ASSET_PATHS.get(request_path)
        if asset is not None:
            return asset if asset.is_file() else None
        if not request_path.startswith(ARTIFACT_URL_PREFIX):
            return None
        relative = request_path[len(ARTIFACT_URL_PREFIX):]
        if not relative or "\\" in relative or not relative.lower().endswith(".json"):
            return None
        try:
            candidate = (ARTIFACT_ROOT / Path(*relative.split("/"))).resolve()
            candidate.relative_to(ARTIFACT_ROOT.resolve())
        except (OSError, ValueError):
            return None
        return candidate if candidate.is_file() else None

    def _serve(self) -> None:
        target = self._route()
        if target is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Viewer route not found")
            return
        try:
            size = target.stat().st_size
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Viewer route not found")
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            with target.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._serve()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self._serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local trace viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ViewerRequestHandler)
    print(f"viewer: http://{args.host}:{args.port}/viewer/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
