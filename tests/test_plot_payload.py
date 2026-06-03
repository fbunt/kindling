"""Regression test for the plot-history round-trip bug.

Plot URLs carry a ?t= cache-buster, so the old _build_done_payload did
Path("plots/plot-001.png?t=123").exists() — always False — and plots were
never re-embedded into conversation history. The fix reads a clean `path`.
"""

from app.chat_loop import ChatTurnResult
from app.routes.chat import _build_done_payload


def test_build_done_payload_embeds_plot_via_clean_path(tmp_path):
    png = tmp_path / "plot-turn-000.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"payload")
    result = ChatTurnResult(
        text="done",
        plots=[
            {
                "url": "/plots/plot-turn-000.png?t=1700000000",
                "name": "my-plot",
                "path": str(png),
            }
        ],
    )

    payload = _build_done_payload(result, image_info=None)

    assert payload["plot_images"], "plot must be embedded into history via clean path"
    assert payload["plot_images"][0]["name"] == "my-plot"
    assert payload["plot_images"][0]["mime"] == "image/png"


def test_build_done_payload_url_with_query_string_is_not_a_filesystem_path(tmp_path):
    """Guards the root cause: the ?t= URL is not a usable on-disk path."""
    from pathlib import Path

    png = tmp_path / "plot-turn-001.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    url = "/plots/plot-turn-001.png?t=1700000000"
    assert not Path(url.lstrip("/")).exists()  # the old (buggy) behavior
    assert Path(url.split("?")[0].lstrip("/")).name == png.name
