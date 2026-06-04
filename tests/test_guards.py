from unittest.mock import MagicMock

from app.guards import judge_code


def _client(text: str) -> MagicMock:
    c = MagicMock()
    c.models.generate_content.return_value.text = text
    return c


def test_judge_code_allows_benign():
    allow, reason = judge_code(
        "result = lf.group_by('year').agg(pl.len()).collect()",
        _client('{"allow": true, "reason": "data analysis"}'),
    )
    assert allow is True


def test_judge_code_blocks_malicious():
    allow, reason = judge_code(
        "import socket; socket.create_connection(('evil', 80))",
        _client('{"allow": false, "reason": "opens a network connection"}'),
    )
    assert allow is False
    assert "network" in reason


def test_judge_code_fails_open_on_error():
    c = MagicMock()
    c.models.generate_content.side_effect = RuntimeError("api down")
    allow, reason = judge_code("result = 1", c)
    assert allow is True  # container is the boundary; never block on judge failure


def test_judge_code_fails_open_on_bad_json():
    allow, _ = judge_code("result = 1", _client("not json"))
    assert allow is True


def test_judge_code_defaults_allow_when_field_missing():
    allow, _ = judge_code("result = 1", _client('{"reason": "no allow field"}'))
    assert allow is True
