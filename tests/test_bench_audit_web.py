"""bench audit-web: the review page's API against real audit files."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from bench.audit import agreement, load_audit
from bench.audit_web import WRITE_HEADER, make_server
from bench.grading import response_sha
from bench.io import atomic_write_json

RESPONSE = "**42** acres\n\n| a | b |\n|---|---|\n| 1 | 2 |"


def _entry(qid, trial, **kw):
    return {
        "question_id": qid,
        "trial": trial,
        "trace_path": None,
        "question": f"question {qid}",
        "expected": 42,
        "response": RESPONSE,
        "response_sha": response_sha({"text": RESPONSE}),
        "extraction": {"status": "answer", "value": 42},
        "grader_verdict": True,
        "grader_reason": "ok",
        "partial_data": False,
        "human_verdict": "",
        "human_note": "",
        **kw,
    }


@pytest.fixture
def server(tmp_path):
    dirs = []
    for model in ("model-a", "model-b"):
        d = tmp_path / model
        (d / "L01").mkdir(parents=True)
        atomic_write_json(
            d / "L01" / "trial_00.json",
            {
                "question_id": "L01",
                "trial": 0,
                "query_records": [
                    {"code": "result = 1", "error": None, "data": '[{"n": 42}]'},
                    {"code": "boom", "error": "NameError", "data": None},
                ],
            },
        )
        audit = {"seed": 0, "entries": [_entry("L01", 0), _entry("A01", 3)]}
        atomic_write_json(d / "audit.json", audit)
        dirs.append(d)
    srv = make_server(dirs, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, dirs
    srv.shutdown()
    srv.server_close()


def _url(srv, path):
    host, port = srv.server_address[:2]
    return f"http://{host}:{port}{path}"


def _get(srv, path):
    with urllib.request.urlopen(_url(srv, path)) as r:
        return r.status, r.read()


def _post(srv, payload, header=True):
    req = urllib.request.Request(
        _url(srv, "/api/verdict"),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
        | ({WRITE_HEADER: "1"} if header else {}),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_items(server):
    srv, _ = server
    status, page = _get(srv, "/")
    assert status == 200 and b"Bench audit" in page
    status, body = _get(srv, "/api/items")
    items = json.loads(body)["items"]
    assert sorted(i["id"] for i in items) == [
        "model-a/A01/3",
        "model-a/L01/0",
        "model-b/A01/3",
        "model-b/L01/0",
    ]
    # Shuffled across models (a fixed hash order), not grouped by model.
    assert [i["id"] for i in items] == [
        i["id"] for i in json.loads(_get(srv, "/api/items")[1])["items"]
    ]
    l01 = next(i for i in items if i["id"] == "model-a/L01/0")
    assert [q["code"] for q in l01["queries"]] == ["result = 1", "boom"]
    assert l01["queries"][1]["error"] == "NameError"
    # The trace is found by run dir layout when trace_path is unset; a
    # missing trace just has no queries.
    assert next(i for i in items if i["id"] == "model-a/A01/3")["queries"] == []


def test_save_writes_the_audit_file_report_reads(server):
    srv, dirs = server
    status, body = _post(
        srv,
        {"id": "model-b/L01/0", "human_verdict": "fail", "human_note": "wrong year"},
    )
    assert status == 200 and body["human_verdict"] == "fail"
    saved = load_audit(dirs[1])["entries"][0]
    assert (saved["human_verdict"], saved["human_note"]) == ("fail", "wrong year")
    assert load_audit(dirs[0])["entries"][0]["human_verdict"] == ""  # other model
    trace = {"question_id": "L01", "trial": 0, "auto_accurate": True, "text": RESPONSE}
    assert agreement(load_audit(dirs[1]), [trace])["filled"] == 1
    # Clearing works too.
    assert _post(srv, {"id": "model-b/L01/0", "human_verdict": ""})[0] == 200
    assert load_audit(dirs[1])["entries"][0]["human_verdict"] == ""


@pytest.mark.parametrize(
    "payload, header, status",
    [
        ({"id": "model-a/L01/0", "human_verdict": "pass"}, False, 403),
        ({"id": "model-a/L01/0", "human_verdict": "maybe"}, True, 400),
        ({"id": "model-a/L01/0", "human_note": 5}, True, 400),
        ({"id": "nope"}, True, 400),
        ({"id": "model-z/L01/0", "human_verdict": "pass"}, True, 404),
        ({"id": "model-a/L09/0", "human_verdict": "pass"}, True, 404),
        (
            {"id": "model-a/L01/0", "human_verdict": "pass", "response_sha": "old"},
            True,
            409,
        ),
    ],
)
def test_save_refuses(server, payload, header, status):
    srv, dirs = server
    before = (dirs[0] / "audit.json").read_text()
    assert _post(srv, payload, header=header)[0] == status
    assert (dirs[0] / "audit.json").read_text() == before
