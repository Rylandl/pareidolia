from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.rectify import VolumeData, fit_local_chart, grayscale_png
    from backend.acus import (
        fit_acus,
        fit_acus_audit,
        fit_acus_field,
        fit_acus_padding_audit,
    )
    from backend.acus_compute import compute_status
    from backend.block_sheet_volume import load_block_sheet_payload, load_block_volume
    from backend.region import fit_acus_region
    from backend.slab_flake_audit import slab_flake_audit
    from backend.slab_flake_holdout import slab_flake_holdout
    from backend.slab_sheetlets import slab_sheetlet_slice
    from backend.slab_flakes import slab_flake_plane
    from backend.slab_analysis import slab_overview, slab_status
else:
    from .rectify import VolumeData, fit_local_chart, grayscale_png
    from .acus import fit_acus, fit_acus_audit, fit_acus_field, fit_acus_padding_audit
    from .acus_compute import compute_status
    from .block_sheet_volume import load_block_sheet_payload, load_block_volume
    from .region import fit_acus_region
    from .slab_flake_audit import slab_flake_audit
    from .slab_flake_holdout import slab_flake_holdout
    from .slab_sheetlets import slab_sheetlet_slice
    from .slab_flakes import slab_flake_plane
    from .slab_analysis import slab_overview, slab_status


class RectifierHandler(BaseHTTPRequestHandler):
    volume: VolumeData
    slab_root = Path(os.environ.get("ACUS_SLAB_ROOT", "work/cross-scroll-analysis"))
    server_version = "RectifierPilot/0.1"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Expose-Headers",
            "X-Volume-Shape-XYZ, X-Volume-Origin-XYZ, X-Volume-Extent-XYZ, "
            "X-Volume-Stride, X-Volume-Percentiles",
        )
        self.send_header("Cache-Control", "no-store")

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _binary(
        self,
        body: bytes,
        content_type: str,
        headers: dict[str, str] | None = None,
        status: int = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._json({"error": message}, status)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(
                {"status": "ok", "volume": self.volume.name, "compute": compute_status()}
            )
            return
        if parsed.path == "/api/volume":
            self._json(self.volume.metadata())
            return
        if parsed.path == "/api/slab/status":
            try:
                self._json(slab_status(self.slab_root))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/slab/overview":
            try:
                query = parse_qs(parsed.query)
                maximum_cells = int(query.get("maxCells", ["60000"])[0])
                z_index_value = query.get("zIndex", [None])[0]
                z_index = int(z_index_value) if z_index_value is not None else None
                self._json(
                    slab_overview(
                        self.slab_root,
                        int(max(1000, min(maximum_cells, 100000))),
                        z_index,
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/slab/flakes":
            try:
                query = parse_qs(parsed.query)
                z_index = int(query.get("zIndex", ["-1"])[0])
                maximum_flakes = int(query.get("maxFlakes", ["3"])[0])
                self._json(
                    slab_flake_plane(
                        self.slab_root,
                        z_index,
                        int(max(1, min(maximum_flakes, 5))),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/slab/flake-audit":
            try:
                query = parse_qs(parsed.query)
                z_index = int(query.get("zIndex", ["-1"])[0])
                repetitions = int(query.get("repetitions", ["4"])[0])
                self._json(
                    slab_flake_audit(
                        self.slab_root,
                        z_index,
                        int(max(2, min(repetitions, 8))),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/slab/flake-holdout":
            try:
                query = parse_qs(parsed.query)
                z_index = int(query.get("zIndex", ["-1"])[0])
                repetitions = int(query.get("repetitions", ["4"])[0])
                self._json(
                    slab_flake_holdout(
                        self.slab_root,
                        z_index,
                        int(max(2, min(repetitions, 8))),
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/slab/sheetlets":
            try:
                query = parse_qs(parsed.query)
                z_index = int(query.get("zIndex", ["-1"])[0])
                spacing = int(query.get("spacing", ["64"])[0])
                cell_step = max(2, min(4, round(spacing / 32)))
                self._json(slab_sheetlet_slice(self.slab_root, z_index, cell_step, 4))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/block/sheets":
            try:
                self._json(load_block_sheet_payload())
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/block/volume":
            try:
                query = parse_qs(parsed.query)
                stride = int(query.get("stride", ["2"])[0])
                body, metadata = load_block_volume(stride=stride)
                self._binary(
                    body,
                    "application/octet-stream",
                    {
                        "X-Volume-Shape-XYZ": ",".join(
                            str(value) for value in metadata["shapeXYZ"]
                        ),
                        "X-Volume-Origin-XYZ": ",".join(
                            str(value) for value in metadata["originXYZ"]
                        ),
                        "X-Volume-Extent-XYZ": ",".join(
                            str(value) for value in metadata["extentXYZ"]
                        ),
                        "X-Volume-Stride": str(metadata["stride"]),
                        "X-Volume-Percentiles": ",".join(
                            str(value) for value in metadata["percentiles"]
                        ),
                    },
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if parsed.path == "/api/slice":
            query = parse_qs(parsed.query)
            try:
                axis = query.get("axis", [""])[0]
                index = int(query.get("index", ["-1"])[0])
                image = self.volume.slice(axis, index)
                body = grayscale_png(image)
            except (ValueError, IndexError) as exc:
                self._error(str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/cube":
            query = parse_qs(parsed.query)
            try:
                seed = tuple(int(query.get(axis, ["-1"])[0]) for axis in ("x", "y", "z"))
                size = int(query.get("size", ["64"])[0])
                cube, origin = self.volume.cube(seed, size)
                body = cube.tobytes(order="C")
            except (ValueError, IndexError) as exc:
                self._error(str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Cube-Size", str(size))
            self.send_header("X-Cube-Origin-XYZ", ",".join(str(value) for value in origin))
            self.send_header("X-Cube-Order", "ZYX")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        self._error("not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {
            "/api/fit",
            "/api/needles",
            "/api/field",
            "/api/audit",
            "/api/padding-audit",
            "/api/region",
        }:
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("fit request body is empty or too large")
            request = json.loads(self.rfile.read(length))
            if parsed.path == "/api/region":
                result = fit_acus_region(self.volume, request)
            elif parsed.path == "/api/padding-audit":
                result = fit_acus_padding_audit(self.volume, request)
            elif parsed.path == "/api/audit":
                result = fit_acus_audit(self.volume, request)
            elif parsed.path == "/api/field":
                result = fit_acus_field(self.volume, request)
            elif parsed.path == "/api/needles":
                result = fit_acus(self.volume, request)
            else:
                result = fit_local_chart(self.volume, request)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._error(str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)
            return
        except Exception as exc:  # keep the interactive pilot alive and report the fit failure
            self._error(f"analysis failed: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json(result)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[rectifier] " + (fmt % args) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the phase-neutral local rectification pilot")
    parser.add_argument("--volume", help="Optional 3D ZYX .npy volume; synthetic scroll when omitted")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()
    RectifierHandler.volume = VolumeData.load(args.volume)
    server = ThreadingHTTPServer((args.host, args.port), RectifierHandler)
    print(
        f"Rectifier backend: http://{args.host}:{args.port} "
        f"({RectifierHandler.volume.name}, shape={RectifierHandler.volume.array.shape})",
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
