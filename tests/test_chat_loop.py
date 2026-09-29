"""run_chat_turn tool-budget exhaustion, against a fake genai client."""

from google.genai import types

import app.chat_loop as chat_loop
from app.chat_loop import DoneEvent, run_chat_turn


class _FakeModels:
    """Returns a function call on every round, and plain text once tools are
    disabled; records the (contents, config) of every call."""

    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((list(contents), config))
        fcc = config.tool_config and config.tool_config.function_calling_config
        if fcc and fcc.mode == types.FunctionCallingConfigMode.NONE:
            part = types.Part(text="final answer")
        else:
            part = types.Part(
                function_call=types.FunctionCall(name="get_dataset_info", args={})
            )
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(role="model", parts=[part]))
            ]
        )


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


async def test_exhausted_loop_final_call_disables_tools(monkeypatch):
    async def fake_exec(name, args, client, model, session):
        return '{"columns": []}', []

    monkeypatch.setattr(chat_loop, "execute_function_call_async", fake_exec)
    client = _FakeClient()
    config = types.GenerateContentConfig(
        system_instruction="sys", tools=[types.Tool(function_declarations=[])]
    )
    contents = [types.Content(role="user", parts=[types.Part(text="q")])]

    events = [
        e
        async for e in run_chat_turn(
            client, "m", contents, config, session=None, max_rounds=2
        )
    ]

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.result.loop_exhausted is True
    assert done.result.text == "final answer"

    calls = client.models.calls
    assert len(calls) == 3  # 2 tool rounds + the forced final call
    for _, cfg in calls[:2]:
        assert cfg is config
    final_contents, final_config = calls[-1]
    assert final_config is not config
    fcc = final_config.tool_config.function_calling_config
    assert fcc.mode == types.FunctionCallingConfigMode.NONE
    assert final_config.system_instruction == "sys"
    assert final_config.tools == config.tools
    # The shared config object is not mutated.
    assert config.tool_config is None
    # The final call ends with the budget-exhausted user instruction.
    last = final_contents[-1]
    assert last.role == "user"
    assert "exhausted" in last.parts[0].text
    assert final_contents[-2].parts[0].function_response is not None


async def test_empty_final_response_keeps_fallback_text(monkeypatch):
    async def fake_exec(name, args, client, model, session):
        return "{}", []

    monkeypatch.setattr(chat_loop, "execute_function_call_async", fake_exec)
    client = _FakeClient()
    orig = client.models.generate_content

    def gen(*, model, contents, config):
        resp = orig(model=model, contents=contents, config=config)
        if config.tool_config is not None:
            return types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(parts=[]))]
            )
        return resp

    client.models.generate_content = gen
    config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[])])
    contents = [types.Content(role="user", parts=[types.Part(text="q")])]
    events = [
        e
        async for e in run_chat_turn(
            client, "m", contents, config, session=None, max_rounds=1
        )
    ]
    assert events[-1].result.text.startswith("I ran out of tool-use rounds")


async def test_malformed_function_call_candidate_does_not_crash():
    """finish_reason=MALFORMED_FUNCTION_CALL comes back with content=None; the
    turn must end with the fallback text instead of an AttributeError."""

    class _Malformed:
        def generate_content(self, *, model, contents, config):
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=None, finish_reason="MALFORMED_FUNCTION_CALL"
                    )
                ]
            )

    client = _FakeClient()
    client.models = _Malformed()
    contents = [types.Content(role="user", parts=[types.Part(text="q")])]
    events = [
        e
        async for e in run_chat_turn(
            client, "m", contents, types.GenerateContentConfig(), session=None
        )
    ]
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.result.text.startswith("I ran out of tool-use rounds")
    assert done.result.tool_calls == []


async def test_run_query_without_code_returns_error():
    import json

    from app.tools import execute_function_call_async

    class _NoSession:
        async def run_query(self, code):
            raise AssertionError("must not reach the sandbox")

    for args in ({}, {"code": ""}, {"code": None}):
        result, plots = await execute_function_call_async(
            "run_query", args, client=None, model="m", session=_NoSession()
        )
        assert "error" in json.loads(result)
        assert plots == []
