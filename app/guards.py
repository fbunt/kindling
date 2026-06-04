"""LLM safety guards — defense-in-depth layers on top of the container boundary.

These are NOT the security boundary (the sandbox container is). They are
probabilistic filters that reduce the rate of clearly-malicious code/prompts
reaching the box, and produce logging signal. Both FAIL OPEN on error: if the
judge model is unavailable we proceed, because the container already contains
arbitrary code.
"""

import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gemini-3.1-flash-lite-preview"

_CODE_JUDGE_INSTRUCTION = """\
You review Python code before it runs in a sandboxed data-analysis tool.

The code executes in a locked-down container: no network, read-only filesystem
(only /tmp is writable and ephemeral), no secrets, non-root, destroyed after the
turn. It analyzes a public wildfire dataset that is fully queryable by design.

ALLOW any legitimate data analysis, statistics, modeling, or visualization —
including pandas/polars/numpy/scipy/scikit-learn/matplotlib use, writing scratch
files to /tmp, and ordinary imports (os, json, etc. used for analysis).

BLOCK only code whose evident purpose is something OTHER than analyzing the
dataset — e.g. attempts to break out of the sandbox or attack the host/kernel,
probe or read sensitive host paths, open network connections, spawn shells for
non-analysis ends, or deliberately exhaust resources.

When uncertain, ALLOW — the container already contains the code; your job is to
catch obvious abuse, not to restrict analysis. Respond ONLY with JSON:
{"allow": true|false, "reason": "<short explanation>"}"""


def _judge(system_instruction: str, content: str, client: genai.Client) -> dict:
    resp = client.models.generate_content(
        model=JUDGE_MODEL,
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
        ),
    )
    data = json.loads(resp.text)
    return {
        "allow": bool(data.get("allow", True)),
        "reason": str(data.get("reason", "")),
    }


def judge_code(code: str, client: genai.Client) -> tuple[bool, str]:
    """Return (allow, reason) for a generated run_query code block. Fails open."""
    try:
        v = _judge(_CODE_JUDGE_INSTRUCTION, f"Review this code:\n\n{code}", client)
        return v["allow"], v["reason"]
    except Exception as e:  # noqa: BLE001 — fail open; container is the boundary
        logger.warning("code-judge unavailable, allowing: %s", e)
        return True, f"judge unavailable: {e}"
