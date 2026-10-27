"""web_search's inner Gemini call, against a fake genai client."""

import json

from google.genai import types

from app.tools import FIRE_DATA_TOOLS, _web_search, execute_function_call_async


class _FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.models = _FakeModels(response)


def _response(*, grounding=None, url_meta=None, text="answer"):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                grounding_metadata=grounding,
                url_context_metadata=url_meta,
            )
        ]
    )


def _chunk(uri, title):
    return types.GroundingChunk(web=types.GroundingChunkWeb(uri=uri, title=title))


def test_inner_call_has_search_and_url_context():
    client = _FakeClient(_response())
    _web_search("read https://example.org/paper.pdf", client, "m")
    model, contents, config = client.models.calls[0]
    assert model == "m"
    assert contents == "read https://example.org/paper.pdf"
    assert any(t.google_search for t in config.tools)
    assert any(t.url_context for t in config.tools)


def test_returns_sources_once_each_and_url_statuses():
    grounding = types.GroundingMetadata(
        grounding_chunks=[
            _chunk("https://a.example/1", "a.example"),
            _chunk("https://a.example/1", "a.example"),
            _chunk("https://b.example/2", "b.example"),
        ]
    )
    url_meta = types.UrlContextMetadata(
        url_metadata=[
            types.UrlMetadata(
                retrieved_url="https://example.org/paper.pdf",
                url_retrieval_status=types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
            ),
            types.UrlMetadata(
                retrieved_url="https://journal.example/x",
                url_retrieval_status=types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_PAYWALL,
            ),
        ]
    )
    out = _web_search(
        "q", _FakeClient(_response(grounding=grounding, url_meta=url_meta)), "m"
    )
    assert out["result"] == "answer"
    assert out["sources"] == [
        {"title": "a.example", "uri": "https://a.example/1"},
        {"title": "b.example", "uri": "https://b.example/2"},
    ]
    assert out["urls_read"] == [
        {"url": "https://example.org/paper.pdf", "status": "success"},
        {"url": "https://journal.example/x", "status": "paywall"},
    ]


def test_no_metadata_returns_text_only():
    assert _web_search("q", _FakeClient(_response()), "m") == {"result": "answer"}
    empty = types.GenerateContentResponse(candidates=[])
    assert _web_search("q", _FakeClient(empty), "m") == {"result": None}


async def test_dispatcher_returns_sources_to_the_model():
    grounding = types.GroundingMetadata(
        grounding_chunks=[_chunk("https://a.example/1", "a")]
    )
    client = _FakeClient(_response(grounding=grounding))
    result, plots = await execute_function_call_async(
        "web_search", {"query": "q"}, client=client, model="m", session=None
    )
    assert json.loads(result)["sources"] == [
        {"title": "a", "uri": "https://a.example/1"}
    ]
    assert plots == []


def test_declaration_tells_the_model_it_can_read_urls():
    (decl,) = [
        d for d in FIRE_DATA_TOOLS.function_declarations if d.name == "web_search"
    ]
    assert "URL" in decl.description


def _read(url, status):
    return types.UrlMetadata(
        retrieved_url=url,
        url_retrieval_status=getattr(
            types.UrlRetrievalStatus, f"URL_RETRIEVAL_STATUS_{status.upper()}"
        ),
    )


def test_inner_call_is_told_to_read_urls():
    client = _FakeClient(_response())
    _web_search("q", client, "m")
    instruction = client.models.calls[0][2].system_instruction
    assert "URL" in instruction and "memory" in instruction


def test_unread_url_gets_a_note():
    out = _web_search(
        "What does https://arxiv.org/pdf/1706.03762 propose?",
        _FakeClient(_response()),
        "m",
    )
    assert "Not read: https://arxiv.org/pdf/1706.03762." in out["note"]


def test_failed_read_gets_a_note():
    url_meta = types.UrlContextMetadata(
        url_metadata=[_read("https://link.springer.com/article/x", "error")]
    )
    out = _web_search(
        "Summarize https://fireecology.example/x.",
        _FakeClient(_response(url_meta=url_meta)),
        "m",
    )
    # the sentence's full stop is not part of the URL
    assert out["note"].startswith("Not read: https://fireecology.example/x. This")


def test_read_url_and_redirect_get_no_note():
    url_meta = types.UrlContextMetadata(
        url_metadata=[_read("https://arxiv.org/abs/2510.01234/", "success")]
    )
    out = _web_search(
        "Title of https://arxiv.org/abs/2510.01234 ?",
        _FakeClient(_response(url_meta=url_meta)),
        "m",
    )
    assert "note" not in out
    redirected = types.UrlContextMetadata(
        url_metadata=[_read("https://final.example/page", "success")]
    )
    out = _web_search(
        "Read https://short.example/p", _FakeClient(_response(url_meta=redirected)), "m"
    )
    assert "note" not in out


def test_query_without_url_gets_no_note():
    assert "note" not in _web_search(
        "largest fire in 2021", _FakeClient(_response()), "m"
    )
