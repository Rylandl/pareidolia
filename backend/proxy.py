from __future__ import annotations

import argparse
import http.client
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class LoopbackProxyHandler(BaseHTTPRequestHandler):
    upstream_host = "127.0.0.1"
    upstream_port = 3000
    api_host = "127.0.0.1"
    api_port = 8000
    protocol_version = "HTTP/1.1"
    server_version = "RectifierTailnetProxy/0.1"

    def _forward(self) -> None:
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.send_error(HTTPStatus.UPGRADE_REQUIRED, "WebSocket HMR is unavailable through the pilot proxy")
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        is_api = self.path == "/health" or self.path.startswith("/api/")
        target_host = self.api_host if is_api else self.upstream_host
        target_port = self.api_port if is_api else self.upstream_port
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP
            and key.lower() not in {"host", "content-length", "origin", "referer"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{target_host}:{target_port}"
        if not is_api:
            headers["Origin"] = f"http://{self.upstream_host}:{self.upstream_port}"
            headers["Referer"] = f"http://{self.upstream_host}:{self.upstream_port}/"
            headers["Sec-Fetch-Site"] = "same-origin"
        timeout = 300 if self.path.startswith(
            ("/api/region", "/api/slab/overview", "/api/block/volume")
        ) else 30
        connection = http.client.HTTPConnection(target_host, target_port, timeout=timeout)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = b"" if self.command == "HEAD" else response.read()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP and key.lower() != "content-length":
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except (ConnectionError, TimeoutError, OSError) as exc:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"local UI is not ready: {exc}")
        finally:
            connection.close()

    do_GET = _forward  # noqa: N815
    do_HEAD = _forward  # noqa: N815
    do_POST = _forward  # noqa: N815
    do_OPTIONS = _forward  # noqa: N815

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[tailnet-ui] " + (fmt % args) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose the loopback vinext server on one tailnet address")
    parser.add_argument("--host", required=True, help="Tailscale address to bind")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--upstream", default="http://127.0.0.1:3000")
    parser.add_argument("--api-upstream", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    upstream = urlsplit(args.upstream)
    if upstream.scheme != "http" or not upstream.hostname:
        raise ValueError("upstream must be an http URL")
    LoopbackProxyHandler.upstream_host = upstream.hostname
    LoopbackProxyHandler.upstream_port = upstream.port or 80
    api_upstream = urlsplit(args.api_upstream)
    if api_upstream.scheme != "http" or not api_upstream.hostname:
        raise ValueError("api-upstream must be an http URL")
    LoopbackProxyHandler.api_host = api_upstream.hostname
    LoopbackProxyHandler.api_port = api_upstream.port or 80
    server = ThreadingHTTPServer((args.host, args.port), LoopbackProxyHandler)
    print(
        f"Rectifier tailnet proxy: http://{args.host}:{args.port} "
        f"-> ui {args.upstream}, api {args.api_upstream}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
