"""Local web page for filling the hand audit (`python -m bench audit-web`).

Serves every audit.json under the given run dirs on 127.0.0.1 as one review
queue: the response rendered as markdown (below the scripted earlier turns,
for a question asked with a history), the expected value, the queries the
model ran with their results, and the plots copied into the run dir. A plot
is served by an opaque entry key (a keyed hash of the entry id, so its URL
does not name the model) and an index, from the path recorded in that audit
entry, resolved under its run dir; a path that leads outside it is refused.

Judging is blind by default: entries from all models are shuffled together
(a fixed order from a hash of each id); the grader's verdict and the
extraction stay hidden until a verdict is recorded (or revealed on request),
and the model name until every entry sharing that question and trial is
judged (the models share sample slots, so one twin's model would name the
other). Saving writes `human_verdict` and `human_note` straight into that
run's audit.json, so `report` picks them up with no copy step.

Writes need an `X-Bench-Audit` header, which a cross-site form or simple
fetch cannot send without a preflight this server never answers, so another
page open in the browser cannot change verdicts.
"""

import hashlib
import hmac
import json
import logging
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from bench.audit import AUDIT_FILE, AuditFileError, load_audit
from bench.io import atomic_write_json, read_trace

logger = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("audit_web.html")
WRITE_HEADER = "X-Bench-Audit"
VERDICTS = ("", "pass", "fail")
# A query result shown in the page is cut to this many characters.
MAX_DATA_CHARS = 20_000


def _entry_id(model: str, entry: dict) -> str:
    return f"{model}/{entry.get('question_id')}/{entry.get('trial')}"


def _trace_for(run_dir: Path, entry: dict) -> dict | None:
    candidates = []
    if entry.get("trace_path"):
        candidates.append(Path(entry["trace_path"]))
    trial = entry.get("trial")
    if isinstance(trial, int):
        candidates.append(
            run_dir / str(entry.get("question_id")) / f"trial_{trial:02d}.json"
        )
    for path in candidates:
        if path.exists():
            trace, _ = read_trace(path)
            if trace is not None:
                return trace
    return None


def _queries(trace: dict | None) -> list[dict]:
    out = []
    for r in (trace or {}).get("query_records") or []:
        data = r.get("data")
        if isinstance(data, str) and len(data) > MAX_DATA_CHARS:
            data = data[:MAX_DATA_CHARS]
            cut = True
        else:
            cut = bool(r.get("data_capped"))
        out.append(
            {
                "code": r.get("code") or "",
                "error": r.get("error"),
                "data": data,
                "data_cut": cut,
                "total_rows": r.get("total_rows"),
                "truncated": r.get("truncated"),
            }
        )
    return out


class AuditStore:
    """The audit files of one or more run dirs, read fresh on every request
    so a hand edit made while the page is open is never overwritten."""

    def __init__(self, run_dirs: list[Path]):
        self.run_dirs = {d.name: d for d in run_dirs}
        self.lock = threading.Lock()
        # Plot URLs carry a keyed hash of the entry id, never the id itself:
        # the id names the model, which must stay hidden while judging blind.
        self._plot_secret = secrets.token_bytes(32)

    def plot_key(self, entry_id: str) -> str:
        return hmac.new(
            self._plot_secret, entry_id.encode(), hashlib.sha256
        ).hexdigest()[:32]

    def plot_url(self, entry_id: str, index: int) -> str:
        return f"/api/plot?key={self.plot_key(entry_id)}&index={index}"

    def items(self) -> list[dict]:
        items = []
        for model, run_dir in self.run_dirs.items():
            audit = load_audit(run_dir)
            if audit is None:
                continue
            for e in audit["entries"]:
                trace = _trace_for(run_dir, e)
                items.append(
                    {
                        "id": _entry_id(model, e),
                        "model": model,
                        "question_id": e.get("question_id"),
                        "trial": e.get("trial"),
                        "question": e.get("question"),
                        "expected": e.get("expected"),
                        "response": e.get("response") or "",
                        "response_sha": e.get("response_sha"),
                        "extraction": e.get("extraction"),
                        "grader_verdict": e.get("grader_verdict"),
                        "grader_reason": e.get("grader_reason"),
                        "partial_data": e.get("partial_data"),
                        "history": e.get("history") or [],
                        "plots": [
                            self.plot_url(_entry_id(model, e), i)
                            for i in range(len(e.get("plots") or []))
                        ],
                        "human_verdict": e.get("human_verdict") or "",
                        "human_note": e.get("human_note") or "",
                        "queries": _queries(trace),
                    }
                )
        items.sort(key=lambda i: hashlib.sha256(i["id"].encode()).hexdigest())
        return items

    def plot_file(self, key: str, index: int) -> Path:
        """The PNG behind the index-th recorded plot of the entry whose
        plot_key is `key`. Raises KeyError (no such entry, index or file) or
        PermissionError (the recorded path is not a .png inside the entry's
        run dir)."""
        found = None
        for model, run_dir in self.run_dirs.items():
            for e in (load_audit(run_dir) or {}).get("entries", []):
                if hmac.compare_digest(self.plot_key(_entry_id(model, e)), key):
                    found = run_dir, e
                    break
            if found:
                break
        if found is None:
            raise KeyError(key)
        run_dir, entry = found
        plots = entry.get("plots") or []
        if not 0 <= index < len(plots):
            raise KeyError(key)
        if not isinstance(plots[index], str):
            raise PermissionError(key)
        root = run_dir.resolve()
        path = (root / plots[index]).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".png":
            raise PermissionError(key)
        if not path.is_file():
            raise KeyError(key)
        return path

    def save(self, payload: dict) -> dict:
        """Set one entry's human_verdict / human_note. Raises ValueError (bad
        request), KeyError (unknown entry) or LookupError (the entry's response
        changed since the page loaded)."""
        entry_id = payload.get("id")
        verdict = payload.get("human_verdict", "")
        note = payload.get("human_note", "")
        if not isinstance(entry_id, str) or entry_id.count("/") < 2:
            raise ValueError("id must be '<model>/<question>/<trial>'")
        if verdict not in VERDICTS:
            raise ValueError(f"human_verdict must be one of {VERDICTS}")
        if not isinstance(note, str):
            raise ValueError("human_note must be a string")
        model = entry_id.rsplit("/", 2)[0]
        run_dir = self.run_dirs.get(model)
        if run_dir is None:
            raise KeyError(entry_id)
        with self.lock:
            audit = load_audit(run_dir)
            for e in (audit or {}).get("entries", []):
                if _entry_id(model, e) != entry_id:
                    continue
                seen = payload.get("response_sha")
                if seen and e.get("response_sha") and seen != e["response_sha"]:
                    raise LookupError(entry_id)
                e["human_verdict"] = verdict
                e["human_note"] = note
                atomic_write_json(run_dir / AUDIT_FILE, audit)
                logger.info("audit %s: %s", entry_id, verdict or "(cleared)")
                return {"id": entry_id, "human_verdict": verdict, "human_note": note}
        raise KeyError(entry_id)


def make_handler(store: AuditStore):
    class Handler(BaseHTTPRequestHandler):
        store_ref = store

        def log_message(self, fmt, *args):
            logger.debug("%s " + fmt, self.address_string(), *args)

        def _send(self, status, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj) -> None:
            self._send(
                status, json.dumps(obj, default=str).encode(), "application/json"
            )

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/items":
                try:
                    self._json(HTTPStatus.OK, {"items": store.items()})
                except AuditFileError as e:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(e)})
            elif urlsplit(self.path).path == "/api/plot":
                self._plot(parse_qs(urlsplit(self.path).query))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"detail": "not found"})

        def _plot(self, query: dict) -> None:
            key = (query.get("key") or [""])[0]
            index = (query.get("index") or [""])[0]
            try:
                path = store.plot_file(key, int(index))
                body = path.read_bytes()
            except (KeyError, ValueError, OSError) as e:
                if isinstance(e, PermissionError):
                    self._json(HTTPStatus.FORBIDDEN, {"detail": "plot path refused"})
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"detail": "no such plot"})
                return
            except AuditFileError as e:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(e)})
                return
            self._send(HTTPStatus.OK, body, "image/png")

        def do_POST(self):
            if self.path != "/api/verdict":
                self._json(HTTPStatus.NOT_FOUND, {"detail": "not found"})
                return
            if self.headers.get(WRITE_HEADER) != "1":
                self._json(HTTPStatus.FORBIDDEN, {"detail": f"missing {WRITE_HEADER}"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(min(length, 1_000_000)) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                self._json(HTTPStatus.OK, store.save(payload))
            except LookupError as e:
                status = (
                    HTTPStatus.NOT_FOUND
                    if isinstance(e, KeyError)
                    else HTTPStatus.CONFLICT
                )
                detail = (
                    f"no audit entry {e}"
                    if status == HTTPStatus.NOT_FOUND
                    else f"{e}: the response changed since the page loaded; reload"
                )
                self._json(status, {"detail": detail})
            except (ValueError, AuditFileError) as e:
                self._json(HTTPStatus.BAD_REQUEST, {"detail": str(e)})

    return Handler


def make_server(run_dirs: list[Path], port: int = 8765) -> ThreadingHTTPServer:
    """Bound to loopback only; port 0 picks a free port (tests)."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(AuditStore(run_dirs)))


def serve(run_dirs: list[Path], port: int = 8765) -> None:
    missing = [str(d) for d in run_dirs if not (d / AUDIT_FILE).exists()]
    if len(missing) == len(run_dirs):
        raise AuditFileError(
            f"no {AUDIT_FILE} under {', '.join(map(str, run_dirs))}; "
            "run `bench audit` first"
        )
    for d in missing:
        logger.warning("%s: no %s, skipped", d, AUDIT_FILE)
    server = make_server(run_dirs, port)
    host, bound = server.server_address[:2]
    print(f"audit review at http://{host}:{bound}/  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
