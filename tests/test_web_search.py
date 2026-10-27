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
