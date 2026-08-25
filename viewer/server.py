"""Serve the repository-local debug-trace viewer without third-party packages."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local trace viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    handler = lambda *handler_args: SimpleHTTPRequestHandler(
        *handler_args, directory=str(root)
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"viewer: http://{args.host}:{args.port}/viewer/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
