"""bench audit-web: the review page's API against real audit files."""

import json
import threading
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

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


# ---------------------------------------------------------------------------
# scripted history and plots
# ---------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
HISTORY = [
    {"role": "user", "text": "What were the 5 largest fires of 2020?"},
    {"role": "model", "text": "| Incid_Name |\n|---|\n| AUGUST COMPLEX |"},
]


@pytest.fixture(scope="module")
def plot_server(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("plots")
    run_dir = tmp_path / "model-a"
    (run_dir / "E09" / "plots").mkdir(parents=True)
    (run_dir / "E09" / "plots" / "trial_00-plot-003.png").write_bytes(PNG)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    (run_dir / "E09" / "plots" / "link.png").symlink_to(outside)
    (run_dir / "E09" / "notes.txt").write_text("not a plot")
    plots = [
        "E09/plots/trial_00-plot-003.png",
        "../outside.png",
        str(outside),
        "E09/plots/link.png",
        "E09/plots/gone.png",
        "E09/notes.txt",
    ]
    audit = {
        "seed": 0,
        "entries": [
            _entry("E09", 0, plots=plots, history=[]),
            _entry("E08", 0, plots=[], history=HISTORY),
        ],
    }
    atomic_write_json(run_dir / "audit.json", audit)
    srv = make_server([run_dir], port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _fetch(srv, path):
    try:
        with urllib.request.urlopen(_url(srv, path)) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def _plot_key(srv, qid):
    """The opaque key the server put in model-a's <qid> trial-0 plot URLs."""
    items = json.loads(_get(srv, "/api/items")[1])["items"]
    entry = next(i for i in items if i["id"] == f"model-a/{qid}/0")
    if entry["plots"]:
        return parse_qs(urlsplit(entry["plots"][0]).query)["key"][0]
    # An entry without plots has no URL to read its key from.
    return srv.RequestHandlerClass.store_ref.plot_key(f"model-a/{qid}/0")


def test_items_carry_history_and_plot_urls(plot_server):
    _, page = _get(plot_server, "/")
    assert b"Earlier turns" in page
    items = {
        i["id"]: i for i in json.loads(_get(plot_server, "/api/items")[1])["items"]
    }
    assert items["model-a/E08/0"]["history"] == HISTORY
    assert items["model-a/E08/0"]["plots"] == []
    urls = items["model-a/E09/0"]["plots"]
    assert len(urls) == 6
    assert urls[0] == f"/api/plot?key={_plot_key(plot_server, 'E09')}&index=0"
    # The model stays out of the URL the page puts in each <img src>.
    assert not any("model-a" in u for u in urls)
    # Entries written before history and plots existed have neither.
    assert items["model-a/E09/0"]["history"] == []


def test_plot_route_serves_the_recorded_file(plot_server):
    key = _plot_key(plot_server, "E09")
    status, headers, body = _fetch(plot_server, f"/api/plot?key={key}&index=0")
    assert status == 200 and body == PNG
    assert headers["Content-Type"] == "image/png"
    assert headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize(
    ("query", "status"),
    [
        ("key={E09}&index=1", 403),  # ../ out of the run dir
        ("key={E09}&index=2", 403),  # absolute path outside it
        ("key={E09}&index=3", 403),  # symlink leading outside it
        ("key={E09}&index=4", 404),  # recorded, but no file
        ("key={E09}&index=5", 403),  # not a PNG
        ("key={E09}&index=6", 404),  # no such index
        ("key={E09}&index=-1", 404),
        ("key={E09}&index=x", 404),
        ("key={E08}&index=0", 404),  # entry without plots
        ("key=model-a%2FE09%2F0&index=0", 404),  # a raw entry id is not a key
        ("key=00000000000000000000000000000000&index=0", 404),  # unknown key
        ("key=&index=0", 404),
        ("index=0", 404),
    ],
)
def test_plot_route_refuses(plot_server, query, status):
    query = query.format(
        E09=_plot_key(plot_server, "E09"), E08=_plot_key(plot_server, "E08")
    )
    got, headers, _ = _fetch(plot_server, f"/api/plot?{query}")
    assert got == status
    assert headers["Content-Type"] == "application/json"
