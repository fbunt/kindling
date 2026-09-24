"""Unit tests for the history/plot helpers in app.routes.chat.

Plots travel through conversation history as {name, epoch} references, not
base64. The server re-reads PLOTS_DIR/<name>.png by name (never via the /plots
URL, whose ?t= cache-buster made Path(url) a nonexistent file in an earlier
bug) for the last HISTORY_IMAGE_WINDOW turns and emits text stubs otherwise.
"""

import base64
import json
from pathlib import Path

import pytest

from app.chat_loop import ChatTurnResult
from app.routes import chat
from app.routes.chat import (
    HISTORY_ADAPTER,
    ImageRef,
    PlotRef,
    _build_done_payload,
    _clean_label,
    _decode_history_image,
    _plot_parts,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_build_done_payload_passes_plots_with_epoch_and_no_base64():
    result = ChatTurnResult(
        text="done",
        plots=[
            {
                "url": "/plots/plot-000.png?t=1",
                "name": "plot-000",
                "path": "plots/plot-000.png",
            }
        ],
        queries_run=["df.head()"],
    )
    payload = _build_done_payload(result, model="m")
    assert payload == {
        "response": "done",
        "model": "m",
        "plots": [
            {
                "url": "/plots/plot-000.png?t=1",
                "name": "plot-000",
                "path": "plots/plot-000.png",
                "epoch": chat.PLOT_EPOCH,
            }
        ],
        "queries": ["df.head()"],
    }
    assert "plot_images" not in payload and "image_info" not in payload


def test_build_done_payload_without_plots_or_queries():
    payload = _build_done_payload(ChatTurnResult(text="t"), model="m")
    assert payload == {"response": "t", "model": "m"}


def test_plot_parts_lookup_is_by_name_not_url(tmp_path, monkeypatch):
    monkeypatch.setattr("app.routes.chat.PLOTS_DIR", tmp_path)
    png = PNG_MAGIC + b"bytes"
    (tmp_path / "plot-003.png").write_bytes(png)
    url = "/plots/plot-003.png?t=1700000000"
    assert not Path(url.lstrip("/")).exists()  # the URL is not a filesystem path
    parts = _plot_parts(PlotRef(name="plot-003", epoch=chat.PLOT_EPOCH), in_window=True)
    assert [p.text for p in parts] == ["[Generated plot: plot-003]", None]
    assert parts[1].inline_data.data == png
    assert parts[1].inline_data.mime_type == "image/png"


def test_plot_parts_stub_texts(tmp_path, monkeypatch):
    monkeypatch.setattr("app.routes.chat.PLOTS_DIR", tmp_path)
    (tmp_path / "plot-001.png").write_bytes(PNG_MAGIC)
    ok = PlotRef(name="plot-001", epoch=chat.PLOT_EPOCH)

    elided = _plot_parts(ok, in_window=False)
    assert [p.text for p in elided] == [
        "[Generated plot: plot-001 (image omitted from context)]"
    ]

    missing = _plot_parts(
        PlotRef(name="plot-002", epoch=chat.PLOT_EPOCH), in_window=True
    )
    assert [p.text for p in missing] == [
        "[Generated plot: plot-002 (image no longer available)]"
    ]

    bad_epoch = _plot_parts(PlotRef(name="plot-001", epoch="dead"), in_window=True)
    assert [p.text for p in bad_epoch] == [
        "[Generated plot: plot-001 (image no longer available)]"
    ]
    no_epoch = _plot_parts(PlotRef(name="plot-001"), in_window=True)
    assert [p.text for p in no_epoch] == [p.text for p in bad_epoch]


def test_plot_parts_label_is_cleaned():
    # The schema caps name at 64 chars; control characters are stripped here.
    parts = _plot_parts(PlotRef(name="bad\nname\x00" + "z" * 40), in_window=False)
    (text,) = [p.text for p in parts]
    assert "\n" not in text and "\x00" not in text
    assert (
        text == "[Generated plot: badname" + "z" * 40 + " (image omitted from context)]"
    )
    assert _clean_label("a" * 300 + "\x7f") == "a" * 100


@pytest.mark.parametrize(
    "ref",
    [
        ImageRef(mime="image/png", data="%%%"),
        ImageRef(
            mime="image/png",
            data=base64.b64encode(b"\xff\xd8\xff" + b"\0" * 8).decode(),
        ),
        ImageRef(mime="image/gif", data=base64.b64encode(b"GIF89a").decode()),
        ImageRef(mime="image/png", data=None),
    ],
)
def test_decode_history_image_rejects_bad_input(ref):
    assert _decode_history_image(ref) is None


def test_decode_history_image_accepts_valid_png_and_jpeg():
    png = PNG_MAGIC + b"\0" * 8
    assert (
        _decode_history_image(
            ImageRef(mime="image/png", data=base64.b64encode(png).decode())
        )
        == png
    )
    jpg = b"\xff\xd8\xff\xe0" + b"\0" * 8
    assert (
        _decode_history_image(
            ImageRef(mime="image/jpeg", data=base64.b64encode(jpg).decode())
        )
        == jpg
    )


def test_decode_history_image_size_cap_lives_in_helper_not_schema(monkeypatch):
    monkeypatch.setattr("app.routes.chat._MAX_UPLOAD_BYTES", 16)
    big = base64.b64encode(PNG_MAGIC + b"\0" * 200).decode()
    # The schema accepts it (no max_length on data) ...
    msgs = HISTORY_ADAPTER.validate_json(
        json.dumps(
            [
                {
                    "role": "user",
                    "content": "x",
                    "image": {"mime": "image/png", "data": big},
                }
            ]
        )
    )
    assert msgs[0].image.data == big
    # ... and the helper is what rejects it.
    assert _decode_history_image(msgs[0].image) is None
    # Just under the cap passes.
    small = base64.b64encode(PNG_MAGIC + b"\0" * 8).decode()
    assert _decode_history_image(ImageRef(mime="image/png", data=small)) is not None
