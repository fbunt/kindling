import json
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.chat_loop import DoneEvent, run_chat_turn
from app.query_engine import configure, create_namespace
from app.tools import FIRE_DATA_TOOLS, SYSTEM_INSTRUCTION

load_dotenv()

_RUN_TS = time.strftime("%Y%m%d-%H%M%S")
_PARQUET = Path(os.environ.get("KINDLING_PARQUET", "data/mtbs_pix_data.parquet"))


def pytest_addoption(parser):
    parser.addoption(
        "--run-evals",
        action="store_true",
        default=False,
        help="Run model-behavior evals (costs real API calls).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-evals"):
        return
    skip_eval = pytest.mark.skip(reason="needs --run-evals")
    for item in items:
        if "model_eval" in item.keywords:
            item.add_marker(skip_eval)


@pytest.fixture(scope="session")
def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def genai_client(api_key) -> genai.Client:
    return genai.Client(api_key=api_key)


@pytest.fixture(scope="session", autouse=True)
def _configure_parquet():
    """Load the parquet once per session so run_query tool calls succeed."""
    if not _PARQUET.exists():
        pytest.skip(f"Parquet not found at {_PARQUET}")
    configure(_PARQUET)


@pytest.fixture
def run_dir(request) -> Path:
    safe_name = request.node.name.replace("/", "_").replace("[", "_").replace("]", "")
    d = Path(".eval-runs") / _RUN_TS / safe_name
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def run_turn(genai_client, run_dir):
    """Returns an async callable that runs one chat turn and dumps a JSON trace."""
    model = os.environ.get("KINDLING_EVAL_MODEL", "gemini-3.1-pro-preview")

    async def _run(prompt: str, trial: int):
        contents = [
            types.Content(role="user", parts=[types.Part(text=prompt)]),
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[FIRE_DATA_TOOLS],
        )
        namespace = create_namespace()
        result = None
        async for ev in run_chat_turn(
            genai_client, model, contents, config, namespace, max_rounds=15
        ):
            if isinstance(ev, DoneEvent):
                result = ev.result
        assert result is not None, "chat turn ended without DoneEvent"

        trace = {
            "prompt": prompt,
            "trial": trial,
            "model": model,
            "tool_calls": result.tool_calls,
            "queries_run": result.queries_run,
            "rejected_queries": result.rejected_queries,
            "loop_exhausted": result.loop_exhausted,
            "text": result.text,
        }
        (run_dir / f"trial_{trial:02d}.json").write_text(
            json.dumps(trace, indent=2, default=str)
        )
        return result

    return _run
