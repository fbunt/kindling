import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from google.genai import types

from app.tools import execute_function_call_async

logger = logging.getLogger(__name__)


@dataclass
class ChatTurnResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    queries_run: list[str] = field(default_factory=list)
    rejected_queries: list[dict] = field(default_factory=list)
    plots: list[dict] = field(default_factory=list)
    loop_exhausted: bool = False


@dataclass
class ThinkingEvent:
    pass


@dataclass
class RunningQueryEvent:
    queries: list[str]


@dataclass
class RejectedEvent:
    queries: list[dict]


@dataclass
class DoneEvent:
    result: ChatTurnResult


async def _is_disconnected(on_disconnect: Callable | None) -> bool:
    if on_disconnect is None:
        return False
    res = on_disconnect()
    if isinstance(res, Awaitable):
        return await res
    return bool(res)


async def run_chat_turn(
    client,
    model: str,
    contents: list,
    config,
    session,
    max_rounds: int = 15,
    on_disconnect: Callable | None = None,
):
    """Run a single chat turn through the tool-use loop.

    Yields structured events (ThinkingEvent, RunningQueryEvent, RejectedEvent,
    DoneEvent). Production code wraps these as SSE; tests collect them
    directly. The DoneEvent carries the full ChatTurnResult.

    `session` is a SandboxSession — query execution always runs in a container.
    """
    all_plots: list[dict] = []
    all_queries: list[str] = []
    tool_calls: list[dict] = []
    last_rejection: dict | None = None
    loop_exhausted = False
    response = None

    for round_num in range(max_rounds):
        if await _is_disconnected(on_disconnect):
            return
        yield ThinkingEvent()

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

        parts = response.candidates[0].content.parts if response.candidates else []
        if logger.isEnabledFor(logging.DEBUG):
            part_types = [type(p).__name__ for p in parts]
            logger.debug(f"Round {round_num}: parts={part_types}")
            for p in parts:
                if hasattr(p, "function_call") and p.function_call:
                    logger.debug(
                        f"  function_call: {p.function_call.name}({p.function_call.args})"  # noqa: E501
                    )
                if hasattr(p, "text") and p.text:
                    logger.debug(f"  text: {p.text[:200]}")

        function_calls = response.function_calls
        if not function_calls:
            logger.info(f"Round {round_num}: 0 function calls — done")
            break
        logger.info(
            f"Round {round_num}: {len(function_calls)} function call(s) "
            f"({', '.join(fc.name for fc in function_calls)})"
        )

        contents.append(response.candidates[0].content)

        round_queries = []
        for fc in function_calls:
            tool_calls.append({"name": fc.name, "args": dict(fc.args or {})})
            if fc.name == "run_query" and fc.args and "code" in fc.args:
                round_queries.append(fc.args["code"])

        for q in round_queries:
            logger.info(f"Executing query: {q}")
        yield RunningQueryEvent(queries=round_queries)

        fc_response_parts = []
        rejected_queries: list[dict] = []
        for fc in function_calls:
            result_str, plots = await execute_function_call_async(
                fc.name, fc.args or {}, client, model, session
            )
            logger.debug(f"  {fc.name} result: {result_str[:500]}")
            result_data = json.loads(result_str)
            all_plots.extend(plots)
            fc_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=result_data,
                    )
                )
            )
            if (
                fc.name == "run_query"
                and fc.args
                and "code" in fc.args
                and "error" in result_data
            ):
                rejected_queries.append(
                    {"code": fc.args["code"], "error": result_data["error"]}
                )
        rejected_codes = {q["code"] for q in rejected_queries}
        all_queries.extend(q for q in round_queries if q not in rejected_codes)
        if rejected_queries:
            last_rejection = rejected_queries[-1]
            yield RejectedEvent(queries=rejected_queries)

        contents.append(types.Content(role="user", parts=fc_response_parts))

        if await _is_disconnected(on_disconnect):
            return
    else:
        loop_exhausted = True
        logger.info("Loop exhausted, making final call")
        yield ThinkingEvent()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=config,
        )

    response_text = _extract_text(response)
    if not response_text:
        _log_empty_response(response, all_plots, all_queries)
        response_text = "I ran out of tool-use rounds before producing an answer."
        if last_rejection:
            response_text += (
                f" The last query failed with: `{last_rejection['error']}`. "
                "Try rephrasing your request or simplifying the query."
            )
        else:
            response_text += " Try rephrasing your request."
    else:
        logger.debug(
            f"Final response_text ({len(response_text)} chars): {response_text[:200]}"
        )

    # rejected_queries here = the cumulative list across rounds is not tracked
    # globally; downstream consumers only need the final aggregate count + last
    # rejection. Tests that need the full history can collect RejectedEvent.
    yield DoneEvent(
        ChatTurnResult(
            text=response_text,
            tool_calls=tool_calls,
            queries_run=all_queries,
            rejected_queries=[last_rejection] if last_rejection else [],
            plots=all_plots,
            loop_exhausted=loop_exhausted,
        )
    )


def _extract_text(response) -> str:
    try:
        return response.text or ""
    except Exception:
        logger.warning(
            "response.text failed, extracting text from parts", exc_info=True
        )
        text = ""
        if response.candidates:
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    text += p.text
        return text


def _log_empty_response(response, all_plots: list, all_queries: list) -> None:
    candidates = response.candidates or []
    finish_reason = candidates[0].finish_reason if candidates else "no_candidates"
    parts = candidates[0].content.parts if candidates else []
    part_details = []
    for p in parts:
        if hasattr(p, "function_call") and p.function_call:
            part_details.append(f"function_call({p.function_call.name})")
        elif hasattr(p, "text") and p.text:
            part_details.append(f"text({len(p.text)} chars)")
        else:
            part_details.append(f"empty({type(p).__name__})")
    logger.warning(
        f"Empty response_text. finish_reason={finish_reason}, "
        f"parts={part_details}, "
        f"function_calls={bool(response.function_calls)}, "
        f"all_plots={len(all_plots)}, all_queries={len(all_queries)}"
    )
